from segment_anything import sam_model_registry, SamPredictor
import src.utils as utils
import tensorrt as trt
import torch
import os
import onnxruntime as ort


class InferenceEngine:
    """
    checkpoint path를 통해 ONNX/TensorRT 모델에 맞는 Inference class를 선택   
    Image 선택과 Predict를 포함한 인터페이스 구현     
    """
    def __init__(self, pth_path, trt_model_1: str, trt_model_2=None):
        
        # Checkpoint 파일 존재 여부 확인   
        if not os.path.isfile(pth_path):
            raise FileNotFoundError(f"couldnt find file\n{pth_path}")
        
        # 모델의 용량이 2GB가 넘는 경우, 즉 vit_h 모델일 경우    
        if os.path.getsize(pth_path) > 2000000000:
            # ONNX 또는 TensorRT 모델인지 구분  
            if trt_model_1.endswith(".onnx"):
                self.currentModel = InferOnnx(pth_path, trt_model_1, trt_model_2, "vit_h")
            else:
                self.currentModel = InferSamH(pth_path, trt_model_1, trt_model_2, "vit_h")
        
        # 모델의 용량이 1GB~2GB에 해당하는 경우, 즉 vit_l 모델일 경우  
        elif os.path.getsize(pth_path) > 1000000000:
            if trt_model_1.endswith(".onnx"):
                self.currentModel = InferOnnx(pth_path, trt_model_1, None, "vit_l")
            else:
                self.currentModel = InferSamBL(pth_path, trt_model_1, "vit_l")
        
        # 모델의 용량이 300MB~1GB에 해당하는 경우, 즉 vit_b 모델일 경우  
        elif os.path.getsize(pth_path) > 300000000:
            if trt_model_1.endswith(".onnx"):
                self.currentModel = InferOnnx(pth_path, trt_model_1, None, "vit_b")
            else:
                self.currentModel = InferSamBL(pth_path, trt_model_1, "vit_b")
        else:
            raise Exception("File size does not match any known sam model sizes ")

    def __call__(self, input_image, input_point, input_label):
        """
        모델의 용량을 기준으로 sub-model을 정했다면, 
        추론 수행 후, Mask Postprocessing 결과 반환

        Args:
            input_image : PIL.Image 또는 numpy 배열
            input_point : 포인트 좌표 배열
            input_label : 포인트 라벨 배열
        return:
            output_image : mask가 적용된 이미지 (numpy 배열)
        """
        # sub-model inference class 호출
        masks = self.currentModel.infer(input_image, input_point, input_label)
        # Postprocessing : 마스크를 원본 이미지에 오버레이    
        output_image = utils.postprocess_masks(input_image, masks)
        return output_image


class InferOnnx:
    """
    SAM Image Encoder (.onnx) 추론 수행 (ONNX Runtime 세션 사용)
    SamPredictor에 Image embedding을 입력하여 최종 마스크 예측 task 수행    
    """
    def __init__(self, pth_path, onnx_model_1, onnx_model_2, model_type):
        
        # ONNX 모델 path 
        self.onnx_model_1 = onnx_model_1
        self.onnx_model_2 = onnx_model_2

        # Pytorch SAM 모델 로드 + predictor 생성    
        sam = sam_model_registry[model_type](checkpoint=pth_path)
        self.predictor = SamPredictor(sam.to('cuda'))
        del sam # 메모리 해제

        # ONNX Runtime 세션 초기화 (CUDA Execution Provider)
        self.session_1 = ort.InferenceSession(self.onnx_model_1, providers=['CUDAExecutionProvider'])
        if self.onnx_model_2 is not None:
            self.session_2 = ort.InferenceSession(self.onnx_model_2, providers=['CUDAExecutionProvider'])

    def infer(self, input_image, input_point, input_label):

        # SAM 전처리 : pixel_mean/std 및 size는 고정    
        pixel_mean = torch.tensor([123.675, 116.28, 103.53])
        pixel_std = torch.tensor([58.395, 57.12, 57.375])
        img_size = 1024

        # numpy 형태로 변환 후 ONNX Inference 입력    
        input_for_onnx = utils.preprocess_image(input_image, 1024, "cpu", pixel_mean, pixel_std, img_size).numpy()
        
        # 첫 번째 ONNX 세션 실행    
        if self.onnx_model_2 is not None:
            # vit_h의 경우, 두 파트 합산 : 세션1 -> 세션2 연결
            output = self.session_2.run(None, {
                "intermediate_input": self.session_1.run(None, {"input_1": input_for_onnx}, )[0]}, )[0]
        else:
            # vit_b, vit_l 단일 ONNX 모델    
            output = self.session_1.run(None, {"input_1": input_for_onnx}, )[0]

        # Output dimension 재조정    
        output = output.reshape((1, 256, 64, 64))

        # SamPredictor에 Image embedding 입력 후 Mask predict task 수행    
        self.predictor.set_image(input_image, embeddings=torch.tensor(output).to("cuda"))
        masks, scores, logits = self.predictor.predict(point_coords=input_point, point_labels=input_label,
                                                       multimask_output=False)
        return masks


class InferSamH:
    """
    TensorRT engine 기반으로 vit_h model의 Image encoder 추론 수행    
    SamPredictor에 Image embedding을 입력하여 최종 마스크 예측 task 수행    
    """
    def __init__(self, pth_path, trt_model_1, trt_model_2, model_type):

        # Pytorch SAM 모델 로드 및 Predictor 설정
        sam = sam_model_registry[model_type](checkpoint=pth_path)
        self.predictor = SamPredictor(sam.to('cuda'))
        del sam # 메모리 제거

        # TensorRT 런타임 세션 및 logger 생성
        TRT_LOGGER = trt.Logger()
        runtime = trt.Runtime(TRT_LOGGER)

        # 첫 번째 engine 로드     
        if not os.path.isfile(trt_model_1):
            raise FileNotFoundError(f"Could not find model in path\n{trt_model_1}")
        with open(trt_model_1, "rb") as f:
            serialized_engine1 = f.read()
        engine1 = runtime.deserialize_cuda_engine(serialized_engine1)
        self.context1 = engine1.create_execution_context()

        # 두 번째 engine 로드   
        if not os.path.isfile(trt_model_2):
            raise FileNotFoundError(f"Could not find model in path\n{trt_model_2}")
        with open(trt_model_2, "rb") as f:
            serialized_engine2 = f.read()
        engine2 = runtime.deserialize_cuda_engine(serialized_engine2)
        self.context2 = engine2.create_execution_context()

        # I/O Buffer 및 CUDA Stream 할당    
        self.inputs1, self.outputs1, self.inputs2, self.outputs2, self.bindings1, self.bindings2, self.stream = utils.allocate_buffers_ensemble(
            engine1, engine2, 1)


    def infer(self, input_image, input_data, input_label):
        # 전처리 : pixel_mean/std 및 size는 고정
        pixel_mean = torch.tensor([123.675, 116.28, 103.53])
        pixel_std = torch.tensor([58.395, 57.12, 57.375])
        img_size = 1024
        input_for_onnx = utils.preprocess_image(input_image, 1024, "cpu", pixel_mean, pixel_std, img_size)

        # 첫 번째 engine I/O Buffer에 이미지 로드
        utils.load_img_to_input_buffer(input_for_onnx, pagelocked_buffer=self.inputs1[0].host)
        # engine 간 pipeline 실행
        [output] = utils.do_inference_v2_ensemble(self.context1, self.context2,
                                                  bindings=[self.bindings1, self.bindings2],
                                                  inputs=[self.inputs1, self.inputs2],
                                                  outputs=[self.outputs1, self.outputs2],
                                                  stream=self.stream)
        # Output dimension 재조정   
        output = output.reshape((1, 256, 64, 64))

        # SamPredictor에 Image embedding 주입    
        self.predictor.set_image(input_image, embeddings=torch.tensor(output).to("cuda"))

        # 입력 형태에 따라 box와 point prompt 분기     
        if isinstance(input_data, torch.Tensor):
            input_data = input_data.numpy()
        if input_data.shape[-1] == 4:
            # Bounding box 방식
            input_box = input_data
            input_label = None
            input_point = None
        else:
            # Point 방식
            input_box = None
            input_point = input_data

        # 최종 마스크 예측    
        masks, scores, logits = self.predictor.predict(point_coords=input_point,
                                                       point_labels=input_label,
                                                       box=input_box,
                                                       multimask_output=False)

        return masks


class InferSamBL:
    """
    TensorRT engine 기반으로 vit_b와 vit_l 모델의 Image Encoder 추론 수행   
    SamPredictor에 Image embedding을 입력하여 최종 마스크 예측 수행    
    """
    def __init__(self, pth_path, trt_model, model_type):
        # Pytorch SAM 모델 로드 + predictor 설정
        sam = sam_model_registry[model_type](checkpoint=pth_path)
        self.predictor = SamPredictor(sam.to('cuda'))
        del sam 

        # TensorRT Runtime 세션 및 logger 생성
        TRT_LOGGER = trt.Logger()
        runtime = trt.Runtime(TRT_LOGGER)

        # TensorRT engine 파일 로드
        if not os.path.isfile(trt_model):
            raise FileNotFoundError(f"Could not find model in path\n{trt_model}")
        with open(trt_model, "rb") as f:
            serialized_engine = f.read()

        engine = runtime.deserialize_cuda_engine(serialized_engine)
        self.context = engine.create_execution_context()

        # I/O Buffer 및 CUDA 스트림 할당
        self.inputs, self.outputs, self.bindings, self.stream = utils.allocate_buffers(engine, max_batch_size=1)

    def infer(self, input_image, input_data, input_label):
        # 전처리 : pixel_mean/std 및 size는 고정    
        pixel_mean = torch.tensor([123.675, 116.28, 103.53])
        pixel_std = torch.tensor([58.395, 57.12, 57.375])
        img_size = 1024
        input_for_onnx = utils.preprocess_image(input_image, 1024, "cpu", pixel_mean, pixel_std, img_size)

        # TensorRT engine에 이미지 로드 후 실행
        utils.load_img_to_input_buffer(input_for_onnx, pagelocked_buffer=self.inputs[0].host)
        [output] = utils.do_inference_v2(self.context, bindings=self.bindings, inputs=self.inputs, outputs=self.outputs,
                                         stream=self.stream)
        # Output dimension 재조정
        output = output.reshape((1, 256, 64, 64))

        # SamPredictor에 Image embedding 주입  
        self.predictor.set_image(input_image, embeddings=torch.tensor(output).to("cuda"))

        # 입력 형태에 따라 box와 point prompt 분기 
        if isinstance(input_data, torch.Tensor):
            input_data = input_data.numpy()
        if input_data.shape[-1] == 4:
            # Bounding Box 방식
            input_box = input_data
            input_label = None
            input_point = None
        else:
            # Point 방식
            input_box = None
            input_point = input_data

        # 최종 마스크 예측    
        masks, scores, logits = self.predictor.predict(point_coords=input_point,
                                                       point_labels=input_label,
                                                       box=input_box,
                                                       multimask_output=False)

        return masks