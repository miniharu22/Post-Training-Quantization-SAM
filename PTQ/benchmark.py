import datetime
import random
import tensorrt as trt
from prettytable import PrettyTable
from segment_anything import sam_model_registry, SamPredictor
import torch
import onnxruntime as ort
import os
import timeit
import numpy as np
from src.infer import InferSamH, InferSamBL
import src.utils as utils
import cv2

screen_width = 1600
screen_height = 900


def calculate_iou(prediction, ground_truth):
    """
    SamPredictor를 통해 예측한 마스크와 실제 마스크 간의 IOU를 계산
    -> IOU : Interaction over Union

    Args:
        predicton : predict를 통해 얻은 binary mask
        ground_truth : 실제 binary mask
    return:
        iou_score : IOU 점수    
    """
    # 예측 마스크와 실제 마스크의 교집합 및 합집합 계산    
    intersection = np.logical_and(ground_truth, prediction)
    union = np.logical_or(ground_truth, prediction)

    # IOU = 교집합 pixel 수 / 합집합 pixel 수    
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score


class DefaultModel:
    """
    Pytorch SAM 모델의 Image Encoder를 이용한 Inference class
    """
    def __init__(self, pth_path, model_type):
        # sam_model_registry를 통해 SAM 모델 로드   
        sam = sam_model_registry[model_type](checkpoint=pth_path)
        # SamPredictor 생성 후 CUDA로 전송
        self.predictor = SamPredictor(sam.to('cuda'))

    def infer(self, input_image, input_point, input_label):
        """
        SamPredictor를 사용해 직접 Mask predict task 수행    
        """
        # 입력 이미지 설정
        self.predictor.set_image(input_image)
        # 포인트 좌표/라벨로 예측 수행    
        masks, scores, logits = self.predictor.predict(point_coords=input_point, point_labels=input_label,
                                                       multimask_output=False)

        return masks


class AccuracyTester:
    """
    Pytorch/TensorRT 기반의 vit_h, vit_l, vit_b 모델의 IOU Accuracy를 비교    
    """
    def __init__(self, img_dir, model_type, sam_path, show_results, save_results):
        # InferSam 객체 저장   
        self.infer32 = None
        self.default_model = None
        self.table = None
        self.infer16 = None

        # 입력 이미지 directory & parameter
        self.img_dir = img_dir
        self.model_type = model_type
        self.sam_path = sam_path
        self.show_results = show_results    # 화면 출력 여부
        self.save_results = save_results    # 파일 저장 여부

        # 파일 저장 시, 결과 directory 생성    
        if save_results:
            current_datetime = datetime.datetime.now()
            datetime_string = current_datetime.strftime("%Y_%m_%d_%H_%M_%S")
            dir_name = f"accuracy_test/{model_type}/{datetime_string}"
            self.output_dir = dir_name
            os.makedirs(self.output_dir)

    def _get_infer_obj(self, precision):
        """
        FP32/FP16 engine에 맞는 InferSam 객체 반환
        """
        if self.model_type == 'vit_h':
            # vit_h는 두 개의 engine을 사용하므로 두 번에 걸쳐서 반환   
            trt_model_path = f'exported_models/{self.model_type}/model_{precision}_1.engine'
            trt_model_path_2 = f'exported_models/{self.model_type}/model_{precision}_2.engine'
            return InferSamH(self.sam_path, trt_model_path, trt_model_path_2, self.model_type)
        else:
            # vit_l, vit_b는 단일 engine이므로 한번만 반환하면 됨     
            trt_model_path = f'exported_models/{self.model_type}/model_{precision}.engine'
            return InferSamBL(self.sam_path, trt_model_path, self.model_type)

    def _get_img_paths(self):
        """
        이미지 directory 내 모든 파일의 path list 반환
        """
        return [os.path.join(self.img_dir, img_name) for img_name in os.listdir(self.img_dir)]

    def load_image_and_data(self, img_path):
        """
        이미지 로드 및 무작위의 point/label 생성   
        """
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        points = []
        labels = []
        # 1~5개의 랜덤 포인트 생성   
        for i in range(random.randint(1, 6)):
            points.append((random.randint(0, image.shape[1]), random.randint(0, image.shape[0])))
            labels.append(1)    # 단일 label로 지정    
        return image, np.array(points), np.array(labels)

    def test_accuracy(self):
        """
        Pytorch, TensorRT (FP32, FP16) 모델의 IOU 비교 및 결과 출력    
        """
        # prettytable 초기화    
        self.table = PrettyTable()
        self.table.field_names = ["Model", "Minimum IOU", "Mean IOU"]
        self.table.title = f"IOU Comparison for {self.model_type}"

        # 이미지 및 데이터 set
        img_paths = self._get_img_paths()
        iou_scores_32 = []
        iou_scores_16 = []
        
        data_arr = []
        for img_path in img_paths:
            data_arr.append(self.load_image_and_data(img_path))
        output_image_default_arr = []
        output_image_fp32_arr = []
        output_image_fp16_arr = []

        # Pytorch model로 predict 수행    
        self.default_model = DefaultModel(self.sam_path, self.model_type)
        for input_image, input_point, input_label in data_arr:
            output_image_default = self.default_model.infer(input_image, input_point, input_label)
            output_image_default_arr.append(output_image_default)
        del self.default_model
        torch.cuda.empty_cache()

        # TensorRT FP32 model로 predict 수행    
        self.infer32 = self._get_infer_obj('fp32')
        for input_image, input_point, input_label in data_arr:
            output_image32 = self.infer32.infer(input_image, input_point, input_label)
            output_image_fp32_arr.append(output_image32)
        del self.infer32
        torch.cuda.empty_cache()

        # TensorRT FP16 model로 predict 수행    
        self.infer16 = self._get_infer_obj('fp16')
        for input_image, input_point, input_label in data_arr:
            output_image16 = self.infer16.infer(input_image, input_point, input_label)
            output_image_fp16_arr.append(output_image16)
        del self.infer16
        torch.cuda.empty_cache()

        # IOU 계산 및 시각화/저장   
        image_count = 0
        for output_image_default, output_image32, output_image16, (original_image, input_point, _) in zip(
                output_image_default_arr, output_image_fp32_arr, output_image_fp16_arr, data_arr):
            iou_score_32 = calculate_iou(output_image32, output_image_default)
            iou_score_16 = calculate_iou(output_image16, output_image_default)
            
            if self.show_results or self.save_results:
                # 마스크 시각화 : Gray-scale -> RGB 변환    
                original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
                output_image_default = np.transpose(output_image_default * 255, (1, 2, 0)).astype(np.uint8)
                output_image32 = np.transpose(output_image32 * 255, (1, 2, 0)).astype(np.uint8)
                output_image16 = np.transpose(output_image16 * 255, (1, 2, 0)).astype(np.uint8)
                output_image_default = cv2.cvtColor(output_image_default, cv2.COLOR_GRAY2RGB)
                output_image32 = cv2.cvtColor(output_image32, cv2.COLOR_GRAY2RGB)
                output_image16 = cv2.cvtColor(output_image16, cv2.COLOR_GRAY2RGB)

                # 포인트 표시    
                for point in input_point:
                    cv2.circle(original_image, point, 3 * (max(original_image.shape) // 800), (0, 0, 255), -1)
                    cv2.circle(output_image_default, point, 3 * (max(original_image.shape) // 800), (0, 0, 255), -1)
                    cv2.circle(output_image32, point, 3 * (max(original_image.shape) // 800), (0, 0, 255), -1)
                    cv2.circle(output_image16, point, 3 * (max(original_image.shape) // 800), (0, 0, 255), -1)

                # 화면 4분할   
                original_image = cv2.resize(original_image, (screen_width // 2, screen_height // 2))
                output_image_default = cv2.resize(output_image_default, (screen_width // 2, screen_height // 2))
                output_image32 = cv2.resize(output_image32, (screen_width // 2, screen_height // 2))
                output_image16 = cv2.resize(output_image16, (screen_width // 2, screen_height // 2))

                screen = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)

                # 4분할된 화면을 배치    
                screen[:screen_height // 2, :screen_width // 2] = original_image
                screen[:screen_height // 2, screen_width // 2:] = output_image_default
                screen[screen_height // 2:, :screen_width // 2] = output_image32
                screen[screen_height // 2:, screen_width // 2:] = output_image16

                # label text    
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(screen, 'Original', (10, 20), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(screen[:screen_height // 2, screen_width // 2:], 'PyTorch', (10, 20), font, 0.5,
                            (0, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(screen[screen_height // 2:, :screen_width // 2], 'TensorRT_FP32', (10, 20), font, 0.5,
                            (0, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(screen[screen_height // 2:, screen_width // 2:], 'TensorRT_FP16', (10, 20), font, 0.5,
                            (0, 255, 0), 1, cv2.LINE_AA)

                if self.show_results:
                    cv2.imshow("Combined Images", screen)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                elif self.save_results:
                    cv2.imwrite(os.path.join(self.output_dir, f"{image_count}.jpg"), screen)
                    image_count += 1
            iou_scores_32.append(iou_score_32)
            iou_scores_16.append(iou_score_16)

        # min/avg IOU 계산    
        min_iou_score_32 = np.min(iou_scores_32)
        mean_iou_score_32 = np.mean(iou_scores_32)
        min_iou_score_16 = np.min(iou_scores_16)
        mean_iou_score_16 = np.mean(iou_scores_16)

        self.table.add_row([f"{self.model_type} FP32", round(min_iou_score_32, 4), round(mean_iou_score_32, 4)])
        self.table.add_row([f"{self.model_type} FP16", round(min_iou_score_16, 4), round(mean_iou_score_16, 4)])

        print(self.table)


class PerformanceTester:
    """
    Pytorch/TensorRT 기반의 vit_h, vit_l, vit_b 모델의 Performance를 비교    
    """
    def __init__(self, model_type, sam_checkpoint, image=None):
        self.model_type = model_type
        self.sam_checkpoint = sam_checkpoint
        self.device = "cuda"

        # 입력 이미지가 없으면 무작위로 생성   
        if image is None:
            self.input_image = np.random.randint(0, 256, (1080, 1920, 3), dtype='uint8')
        else:
            self.input_image = image

        # SAM 모델 및 predictor 초기화 변수    
        self.sam = None
        self.sam_image_encoder = None

        # 전처리 파라미터
        self.pixel_mean = torch.tensor([123.675, 116.28, 103.53])
        self.pixel_std = torch.tensor([58.395, 57.12, 57.375])
        self.img_size = 1024
        self.input_image_processed = None

        # ONNX/TensorRT Runtime    
        self.ort_session = None
        self.runtime = trt.Runtime(trt.Logger(min_severity=trt.ILogger.ERROR))

        # TensorRT engine context 저장
        self.context = None
        self.context_fp16 = None

        # Performance 결과 table    
        self.performance_table = PrettyTable()
        self.performance_table.field_names = ["Model", "Average FPS", "Average Time (sec)", "Relative FPS",
                                              "Relative Time (%)"]
        self.performance_table.title = f"Performance Comparison for {self.model_type}"

    def add_to_table(self, name, fps, times, base_fps=None, base_times=None):
        """
        측정된 performance data를 table에 추가    
        """
        avg_fps = round(fps, 2)
        avg_time = round(times, 6)
        if base_fps and base_times:
            rel_fps = round(fps / base_fps, 2)
            rel_time = round(times / base_times * 100, 2)
            self.performance_table.add_row([name, avg_fps, avg_time, rel_fps, rel_time])
        else:
            self.performance_table.add_row([name, avg_fps, avg_time, 'NA', 'NA'])

    def print_stats(self):
        """
        performance 결과 table 출력     
        """
        print(self.performance_table)

    def load_pytorch_model(self):
        """
        Pytorch SAM Image encoder 및 predictor 로드   
        """
        self.sam = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint)
        self.sam_image_encoder = SamPredictor(self.sam.to(self.device))
        # 전처리된 입력 이미지 생성 
        self.input_image_processed = utils.preprocess_image(self.input_image, 1024, self.device, self.pixel_mean,
                                                            self.pixel_std, self.img_size)

    def unload_pytorch_model(self):
        """
        Pytorch Model 메모리 제거 및 GPU cache 정리    
        """
        del self.sam
        del self.pixel_mean
        del self.pixel_std
        del self.img_size
        del self.sam_image_encoder
        self.input_image_processed = self.input_image_processed.cpu()
        torch.cuda.empty_cache()

    def load_tensorrt_model_fp32(self):
        """
        TensorRT FP32 engine 로드 및 context 생성    
        """
        engine_path = f"exported_models/{self.model_type}/model_fp32.engine"
        if not os.path.isfile(engine_path):
            raise FileNotFoundError("Model has not been generated yet, build the model first.")
        with open(engine_path, "rb") as f:
            serialized_engine = f.read()
        self.context = self.runtime.deserialize_cuda_engine(serialized_engine).create_execution_context()

    def unload_tensorrt_model_fp32(self):
        """
        FP32 engine context 해제 및 cache 정리    
        """
        self.context = None
        torch.cuda.empty_cache()

    def load_tensorrt_model_fp16(self):
        """
        TensorRT FP16 engine 로드 및 context 생성    
        """
        engine_path_fp16 = f"exported_models/{self.model_type}/model_fp16.engine"
        if not os.path.isfile(engine_path_fp16):
            raise FileNotFoundError(f"Model has not been generated yet, build the model first.\n{engine_path_fp16}")
        with open(engine_path_fp16, "rb") as f:
            serialized_engine_fp16 = f.read()
        self.context_fp16 = self.runtime.deserialize_cuda_engine(serialized_engine_fp16).create_execution_context()

    def unload_tensorrt_model_fp16(self):
        """
        FP32 engine context 해제 및 cache 정리    
        """
        self.context_fp16 = None
        torch.cuda.empty_cache()

    def load_tensorrt_model_ensemble(self):
        """
        vit_h 모델의 Ensemble FP32 engine 2개를 로드, Context list 생성    
        """
        TRT_LOGGER = trt.Logger()
        runtime = trt.Runtime(TRT_LOGGER)

        # 첫 번째 engine 로드
        engine_path1 = f"exported_models/{self.model_type}/model_fp32_1.engine"
        if not os.path.isfile(engine_path1):
            raise FileNotFoundError("model has not been generated yet, build model first")
        with open(engine_path1, "rb") as f:
            serialized_engine1 = f.read()
        engine1 = runtime.deserialize_cuda_engine(serialized_engine1)
        context1 = engine1.create_execution_context()

        # 두 번째 engine 로드   
        engine_path2 = f"exported_models/{self.model_type}/model_fp32_2.engine"
        if not os.path.isfile(engine_path2):
            raise FileNotFoundError("model has not been generated yet, build model first")
        with open(engine_path2, "rb") as f:
            serialized_engine2 = f.read()
        engine2 = runtime.deserialize_cuda_engine(serialized_engine2)
        context2 = engine2.create_execution_context()

        # Context list 저장     
        self.context = [context1, context2]

    def unload_tensorrt_model_ensemble(self):
        """
        Ensemble FP32 engine context 해제 및 cache 정리    
        """
        self.context = None
        torch.cuda.empty_cache()

    def load_tensorrt_model_fp16_ensemble(self):
        """
        vit_h 모델의 Ensemble FP16 engine 2개를 로드, Context list 생성    
        """
        TRT_LOGGER = trt.Logger()
        runtime = trt.Runtime(TRT_LOGGER)

        # 첫 번째 engine 로드
        engine_path1 = f"exported_models/{self.model_type}/model_fp16_1.engine"
        if not os.path.isfile(engine_path1):
            raise FileNotFoundError("model has not been generated yet, build model first")
        with open(engine_path1, "rb") as f:
            serialized_engine1 = f.read()
        engine1 = runtime.deserialize_cuda_engine(serialized_engine1)
        context1 = engine1.create_execution_context()

        # 두 번째 engine 로드 
        engine_path2 = f"exported_models/{self.model_type}/model_fp16_2.engine"
        if not os.path.isfile(engine_path2):
            raise FileNotFoundError("model has not been generated yet, build model first")
        with open(engine_path2, "rb") as f:
            serialized_engine2 = f.read()
        engine2 = runtime.deserialize_cuda_engine(serialized_engine2)
        context2 = engine2.create_execution_context()

        self.context_fp16 = [context1, context2]

    def unload_tensorrt_model_fp16_ensemble(self):
        """
        Ensemble FP16 engine context 해제 및 cache 정리    
        """
        self.context_fp16 = None
        torch.cuda.empty_cache()

    def run_pytorch_model(self):
        """
        PyTorch 모델을 실행하기 위한 래퍼 (set_image 호출)
        """
        original_output = self.sam_image_encoder.set_image(self.input_image)

    def run_tensorrt_model_fp32(self):
        """
        TensorRT FP32 engine을 실행    
        """
        tensorrt_output = \
        utils.do_inference_v2(self.context, bindings=self.bindings, inputs=self.inputs, outputs=self.outputs,
                              stream=self.stream)[0]

    def run_tensorrt_model_fp16(self):
        """
        TensorRT FP16 engine을 실행    
        """
        tensorrt_output = \
        utils.do_inference_v2(self.context_fp16, bindings=self.bindings, inputs=self.inputs, outputs=self.outputs,
                              stream=self.stream)[0]

    def run_tensorrt_model_ensemble(self):
        """
        TensorRT Ensemble(FP32) engine을 순차대로 실행   
        """
        tensorrt_output = \
        utils.do_inference_v2_ensemble(*self.context, bindings=self.bindings, inputs=self.inputs, outputs=self.outputs,
                                       stream=self.stream)[0]

    def run_tensorrt_model_fp16_ensemble(self):
        """
        TensorRT Ensemble(FP16) engine을 순차대로 실행   
        """
        tensorrt_output = utils.do_inference_v2_ensemble(*self.context_fp16, bindings=self.bindings, inputs=self.inputs,
                                                         outputs=self.outputs, stream=self.stream)[0]

    def time_model(self, model, warmup_iters, measure_iters):
        """
        입력 받은 모델의 실행 시간을 측정    
        Args:
            warum_iters : 측정 전 워밍 업 반복 횟수   
            measure_iters : 실제 측정 반복 횟수   
        return:
            avg_fps, avg_time
        """
        # 워밍 업 : GPU 최적화 등에 대비용    
        for _ in range(warmup_iters):
            model()

        # 측정
        start_time = timeit.default_timer()
        for _ in range(measure_iters):
            model()
        end_time = timeit.default_timer()
        elapsed_time = end_time - start_time

        avg_fps = measure_iters / elapsed_time
        avg_time = elapsed_time / measure_iters

        return avg_fps, avg_time

    def test_models(self, warmup_iters=5, measure_iters=50):
        """
        각 모델별 로드->측정->제거 과정을 수행하고 결과를 table에 출력    
        """
        # 내부 테스트 함수 호출
        self.__test_models(warmup_iters, measure_iters)
        # 테스트 이후 GPU cache 정리    
        torch.cuda.empty_cache()

    def __test_models(self, warmup_iters=5, measure_iters=50):
        """
        실제 성능 측정 logic이 포함된 내부 method
        """
        # Pytorch 모델 측정    
        self.load_pytorch_model()
        fps1, times1 = self.time_model(self.run_pytorch_model, warmup_iters, measure_iters)
        self.unload_pytorch_model()
        
        # vit_h 모델은 Ensemble 처리, vit_l/vit_b는 단일 engine 처리    
        if self.model_type != "vit_h":
            # TensorRT FP32 engine 실행    
            self.load_tensorrt_model_fp32()
            # I/O Buffer 할당 및 입력 복사   
            self.inputs, self.outputs, self.bindings, self.stream = utils.allocate_buffers(self.context.engine, 1)
            utils.load_img_to_input_buffer(self.input_image_processed, pagelocked_buffer=self.inputs[0].host)
            fps2, times2 = self.time_model(self.run_tensorrt_model_fp32, warmup_iters, measure_iters)
            self.unload_tensorrt_model_fp32()

            # TensorRT FP16 engine 실행  
            self.load_tensorrt_model_fp16()
            self.inputs, self.outputs, self.bindings, self.stream = utils.allocate_buffers(self.context_fp16.engine, 1)
            utils.load_img_to_input_buffer(self.input_image_processed.half(), pagelocked_buffer=self.inputs[0].host)
            fps3, times3 = self.time_model(self.run_tensorrt_model_fp16, warmup_iters, measure_iters)
            self.unload_tensorrt_model_fp16()
        else:
            # vit_h 모델 : Ensemble FP32
            self.load_tensorrt_model_ensemble()
            inputs1, outputs1, inputs2, outputs2, bindings1, bindings2, stream = utils.allocate_buffers_ensemble(
                self.context[0].engine, self.context[0].engine, 1)
            utils.load_img_to_input_buffer(self.input_image_processed, pagelocked_buffer=inputs1[0].host)
            self.inputs, self.outputs, self.bindings, self.stream = [inputs1, inputs2], [outputs1, outputs2], [
                bindings1, bindings2], stream
            fps2, times2 = self.time_model(self.run_tensorrt_model_ensemble, warmup_iters, measure_iters)
            self.unload_tensorrt_model_ensemble()

            # vit_h 모델 : Ensemble FP16
            self.load_tensorrt_model_fp16_ensemble()
            inputs1, outputs1, inputs2, outputs2, bindings1, bindings2, stream = utils.allocate_buffers_ensemble(
                self.context_fp16[0].engine, self.context_fp16[0].engine, 1)
            utils.load_img_to_input_buffer(self.input_image_processed.half(), pagelocked_buffer=inputs1[0].host)
            self.inputs, self.outputs, self.bindings, self.stream = [inputs1, inputs2], [outputs1, outputs2], [
                bindings1, bindings2], stream
            fps3, times3 = self.time_model(self.run_tensorrt_model_fp16_ensemble, warmup_iters, measure_iters)
            self.unload_tensorrt_model_fp16_ensemble()

        self.add_to_table("PyTorch model", fps1, times1, fps1, times1)
        self.add_to_table("TensorRT model", fps2, times2, fps1, times1)
        self.add_to_table("TensorRT FP16 model", fps3, times3, fps1, times1)
        self.print_stats()