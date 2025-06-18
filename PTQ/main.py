import argparse
import numpy as np
import os
import cv2
from src.infer import InferenceEngine
from src.export import ExportSAM
from src.benchmark import PerformanceTester, AccuracyTester
from src.utils import choose_point


def benchmark(sam_checkpoint, model_type, warmup_iters, measure_iters):
    """
    PerformanceTester를 사용하여 모델의 성능(FPS, 실행시간)을 측정    
    Args:
        sam_checkpoint : Pytorch checkpoint 
        model_type : vit_b, vit_l, vit_h, all
        warmup_iters : 워밍업 반복 횟수   
        measure_iters : 성능 측정 반복 횟수   
    """
    valid_model_types = ['vit_b', 'vit_l', 'vit_h', 'all']

    # Checkpoint 유효성 검사    
    assert (os.path.isfile(sam_checkpoint) or (os.path.isdir(
        sam_checkpoint) and model_type == "all")), f"Checkpoint file does not exist: {sam_checkpoint}"
    # Model type 유효성 검사
    assert model_type in valid_model_types, f"Invalid model_type: {model_type}. It must be one of {valid_model_types}."

    # 디렉터리 내 모든 모델에 대해 측정을 수행    
    if model_type == 'all':
        models = [os.path.join(sam_checkpoint, i) for i in os.listdir(sam_checkpoint) if "sam_vit_" in i]
        
        # 각 모델별로 PerformanceTester 실행    
        for m_type in ['vit_b', 'vit_l', 'vit_h']:
            for model in models:
                if m_type in os.path.split(model)[1]:
                    PerformanceTester(m_type, model).test_models(warmup_iters, measure_iters)
    else:
        PerformanceTester(model_type, sam_checkpoint).test_models(warmup_iters, measure_iters)


def accuracy(image_dir, model_type, sam_checkpoint, show_results, save_results):
    """
    AccuracyTester를 사용하여 IOU 측정   
    Args:
        image_dir : 테스트용 이미지 디렉터리
        model_type : vit_b, vit_l, vit_h, all
        sam_checkpoint : Pytorch Checkpoint file (Path)
        show_results : 화면에 결과 이미지 표시 여부
        save_results : 결과 이미지 파일로 저장 여부
    """
    valid_model_types = ['vit_b', 'vit_l', 'vit_h', 'all']

    # Checkpoint 및 model type 유효성 검사 
    assert (os.path.isfile(sam_checkpoint) or (os.path.isdir(
        sam_checkpoint) and model_type == "all")), f"Checkpoint file does not exist: {sam_checkpoint}"
    assert model_type in valid_model_types, f"Invalid model_type: {model_type}. It must be one of {valid_model_types}."

    # 디렉터리 내 모든 모델에 대해 Accuracy Test 수행    
    if model_type == 'all':
        models = [os.path.join(sam_checkpoint, i) for i in os.listdir(sam_checkpoint) if "sam_vit_" in i]
        for m_type in ['vit_b', 'vit_l', 'vit_h']:
            for model in models:
                if m_type in os.path.split(model)[1]:
                    AccuracyTester(image_dir, m_type, model, show_results, save_results).test_accuracy()
    else:
        # 특정 모델에 대해 Accuracy Test 수행
        AccuracyTester(image_dir, model_type, sam_checkpoint, show_results, save_results).test_accuracy()


def infer(pth_path, model_1, model_2, img_path):
    """
    InferenceEngine을 사용하여 입력 이미지에 대해 Mask Predict 수행
    Args:
        pth_path : SAM Checkpoint File path
        model_1 : engine File 1 path
        model_2 : engine File 2 path (vit_h 전용)
        img_path : 입력 이미지 파일 path
    """
    # 필수 파일 존재 여부 확인
    for file_path in [pth_path, model_1, img_path]:
        assert os.path.isfile(file_path), f"File does not exist: {file_path}"
    if model_2:
        assert os.path.isfile(model_2), f"File does not exist: {model_2}"

    # 포인트 선택 
    input_point = choose_point(img_path)
    input_label = np.array([1])

    # Inference 실행
    inference_engine = InferenceEngine(pth_path, model_1, model_2)
    image = cv2.imread(img_path)
    result = inference_engine(image, input_point, input_label)

    cv2.imshow("result", np.array(result).astype(np.uint8))
    cv2.waitKey(0)


def export(model_path, model_precision):
    """
    ExpostSAM을 사용하여 체크포인트를 ONNX & TensorRT 포맷으로 추출
    Args:
        model_path : SAM checkpoint file path
        model_precision : FP32, FP16, Both
    """
    # Checkpoint 파일 유효성 검사
    assert os.path.isfile(model_path), f"Model file does not exist: {model_path}"
    sam_model = ExportSAM(model_path)
    sam_model(model_precision)


if __name__ == "__main__":
    # ArgumentParser 생성 
    # sub-command 설정
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    subparsers.required = True

    # Benchmark 커맨드  
    parser_benchmark = subparsers.add_parser('benchmark')
    parser_benchmark.add_argument('--sam_checkpoint', type=str, required=True)
    parser_benchmark.add_argument('--model_type', type=str, required=True)
    parser_benchmark.add_argument('--warmup_iters', type=int, default=5)
    parser_benchmark.add_argument('--measure_iters', type=int, default=50)
    parser_benchmark.set_defaults(func=benchmark)

    # Inference 커맨드   
    parser_infer = subparsers.add_parser('infer')
    parser_infer.add_argument('--pth_path', type=str, required=True)
    parser_infer.add_argument('--model_1', type=str, required=True)
    parser_infer.add_argument('--model_2', type=str, default=None)
    parser_infer.add_argument('--img_path', type=str, required=True)
    parser_infer.set_defaults(func=infer)

    # Export (Model Conversion) 커맨드    
    parser_export = subparsers.add_parser('export')
    parser_export.add_argument('--model_path', type=str, required=True)
    parser_export.add_argument('--model_precision', type=str, choices=['fp32', 'fp16', 'both'], required=True)
    parser_export.set_defaults(func=export)

    # Accuracy Test 커맨드   
    parser_accuracy = subparsers.add_parser('accuracy')
    parser_accuracy.add_argument('--image_dir', type=str, required=True)
    parser_accuracy.add_argument('--model_type', type=str, required=True)
    parser_accuracy.add_argument('--sam_checkpoint', type=str, required=True)
    parser_accuracy.add_argument('--show_results', action='store_true', default=False)
    parser_accuracy.add_argument('--save_results', action='store_true', default=False)
    parser_accuracy.set_defaults(func=accuracy)

    # Parsing된 인자들로 그에 대응하는 함수 호출   
    args = parser.parse_args()
    func = vars(args).pop('func')
    # Function에 필요한 인자만 딕셔너리로 전달    
    func(**vars(args))