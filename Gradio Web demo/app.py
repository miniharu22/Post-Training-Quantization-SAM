import os
import cv2
import torch
import logging
import numpy as np
import gradio as gr

from typing import Tuple, Dict, Any, List
from pathlib import Path
from scipy.stats import mode
from einops import repeat
from skimage import color
from skimage.measure import label
from segment_anything import (
    SamAutomaticMaskGenerator,
    build_sam_vit_b,
    build_sam_vit_h,
    build_sam_vit_l,
    SamPredictor,
)


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)



# keypoint 탐지 함수
def find_dots(image: np.ndarray, image_with_keypoints: np.ndarray) -> np.ndarray:
    """
    입력된 이미지와 keypoint가 표시된 이미지를 비교,
    차이가 존재하는 pixel들이 연결된 영역을 찾아 좌표값을 반환    
    Args:
        image : 원본 이미지 배열 (Height x Width x 3)
        imgage_with_keypoints : keypoint가 그려진 이미지 배열 (Height x Width x 3)
    return:
        np.ndarray : 찾은 keypoint들의 (x,y) 좌표 배열 (N x 2)
    """
    # pixel 별 차이를 계산하여 마스크 생성 (차이가 존재하는 pixel에 대해서는 True로 처리)
    diff_mask = np.max(image != image_with_keypoints, axis=-1)
    
    # 생성한 마스크에 대해 라벨링을 진행하여 각 keypoint를 구분   
    diff_labels, num_labels = label(diff_mask, return_num=True)
    points = []
    
    # 라벨링된 각 영역의 무게 중심을 계산, keypoint 좌표로 반환    
    for i in range(1, num_labels+1):
        com = np.argwhere(diff_labels == i).mean(axis=0)
        # numpy 좌표를 predictor의 좌표로 반환하여 저장
        points.append([com[1], com[0]])

    return np.array(points)



# 컬러 마스크 생성 함수   
def create_color_mask(image: np.ndarray, annotations: Dict[str, Any]) -> np.ndarray:
    """
    SAM에서 생성된 Annotation(주석) list를 바탕으로 객체별 마스크를 
    하나의 color map 이미지로 합성   
    Args:
        image : 원본 이미지 배열    
        annotations : SAM Annotation dictionary list
    return:
        np.ndarray : 컬러로 시각화된 마스크 이미지    
    """
    # Annotation 없으면 원본 반환
    if len(annotations) == 0:
        return image
    
    # 면적 순으로 정렬 (큰 객체가 나중에 그려지도록)
    sorted_anns = sorted(annotations, key=(lambda x: x['area']), reverse=True)
    
    # 스택 후, pixel 별 최댓값 인덱스 선택
    mask = np.stack([x["segmentation"] for x in sorted_anns])
    mask = np.argmax(mask, axis=0)

    # 라벨을 컬러맵으로 변환
    color_mask = color.label2rgb(mask, image)
    return color_mask



# RGBA Segment 추출 함수
def extract_rgba_masks(
    image: np.ndarray,
    annotations: Dict[str, Any],
    mask_filter_area: float,
) -> List[np.ndarray]:
    """
    각 Annotation mask 영역을 RGBA 이미지로 분리하여 반환   
    Args:
        imgae : 원본 RGB 이미지 배열    
        annotations : SAM Annotation list
        mask_filter_area : 마스크 필터링에 대한 최소 면적    
    return:
        List[np.ndarray] : RGBA 이미지 segment list
    """
    image_segments = []
    for ann in annotations:
        # 면적에 대한 최소값 만족 시
        if np.sum(ann["segmentation"]) >= mask_filter_area:
            # 알파 채널(투명도) 생성 (0/1 -> 0/255)
            segment = repeat(ann["segmentation"].astype(np.uint8) * 255, "h w -> h w 1")
            image_segment = np.concatenate([image, segment], axis=-1)
            image_segments.append(image_segment)

    return image_segments


# Automatic mask generator 실행 함수  
@torch.no_grad()
def generate(
    predictor: SamPredictor,                
    image: np.ndarray,
    points_per_side: int,
    points_per_batch: int,
    pred_iou_thresh: float,
    stability_score_thresh: float,
    stability_score_offset: float,
    box_nms_thresh: float,
    crop_n_layers: int,
    crop_nms_thresh: float,
    crop_overlap_ratio: float,
    crop_n_points_downscale_factor: float,
    min_mask_region_area: float,
    display_rgba_segments: bool,
    mask_filter_area: float,
    progress=gr.Progress(),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    설정된 파라미터로 SAM automatic mask generator task 수행  
    필요 시, RGBA로 분할된 segment를 반환    
    return:
        color_mask : 컬러 합성 마스크   
        annotation_masks : RGBA segment list
    """
    # Automatic Mask Generator 초기화    
    generator = SamAutomaticMaskGenerator(
        predictor.model,
        points_per_side,
        points_per_batch,
        pred_iou_thresh,
        stability_score_thresh,
        stability_score_offset,
        box_nms_thresh,
        crop_n_layers,
        crop_nms_thresh,
        crop_overlap_ratio,
        crop_n_points_downscale_factor,
        min_mask_region_area=min_mask_region_area,
    )
    # 추론 시작 표시 (0%)
    progress(0, "Running inference")
    # 자동으로 Annotation mask 생성
    annotations = generator.generate(image)

    # 컬러 마스크 시각화 (50%)
    progress(0.5, "Making color mask")
    color_mask = create_color_mask(image, annotations)

    # RGBA segment 추출 (75%)
    annotation_masks = []
    if display_rgba_segments:
        progress(0.75, "Extracting RGBA segments")
        annotation_masks = extract_rgba_masks(image, annotations, mask_filter_area)

    # 완료 표시 (100%) + 결과 반환    
    progress(1, "Returning masks")
    return color_mask, annotation_masks

# 모델 로드 함수   
def load_model(
    name: str,
    device: str,
) -> SamPredictor:
    """
    backbone checkpoint 이름에 따라 ViT 모델 build 후 load
    Args:
        name : checkpoint 파일명 (*.pth)
        device : 'cuda' 또는 'cpu'
    return:
        SamPredictor wrapper Instance
    """
    # 모델 checkpoint path 설정   
    checkpoint_path = os.path.join("models", name)

    # checkpoint 파일명에 따라 load할 모델 type 결정 & 빌드   
    if "vit_b" in name:
        model = build_sam_vit_b(checkpoint_path)
    elif "vit_h" in name:
        model = build_sam_vit_h(checkpoint_path)
    elif "vit_l" in name:
        model = build_sam_vit_l(checkpoint_path)
    else:
        raise ValueError(f"Invalid checkpoint name: {name}")

    model.to(device)
    # 모델 load 완료 로그 출력  
    logger.info(f"Loaded model: {name}")

    return SamPredictor(model)


# keypoint 및 box 시각화 함수    
def display_detected_keypoints(
    image: np.ndarray,
    fg_keypoints: np.ndarray,
    bg_keypoints: np.ndarray,
    bbox: np.ndarray,
):
    """
    ForeGround/Background canvas 차이로 keypoint 수 집계
    box canvas에서 bounding box 좌표 추출   
    return:
        AnnotatedImage 입력 데이터, FG/BG keypoint 개수   
    """
    # Foreground keypoint pixel 영역 감지
    pos_diff_mask = np.max(image != fg_keypoints, axis=-1)
    _, num_positive = label(pos_diff_mask, return_num=True)

    # Background keypoint pixel 영역 감지
    neg_diff_mask = np.max(image != bg_keypoints, axis=-1)
    _, num_negative = label(neg_diff_mask, return_num=True)

    # Box 영역 추출    
    sections = [
        (pos_diff_mask, "Positive"),
        (neg_diff_mask, "Negative"),
    ]

    # Bounding Box 좌표 추출   
    box = detect_boxes(image, bbox)
    if box is not None:
        sections.append((tuple(box), "Bounding Box"))

    return ((image, sections), num_positive, num_negative)


# Box 탐지 함수   
def detect_boxes(image: np.ndarray, image_with_boxes: np.ndarray) -> np.ndarray:
    """
    원본 이미지와 Box 스케치 이미지 간의 pixel 차이를 이용,
    bounding box 좌표 반환    
    return:
        np.ndarray : box 좌표 배열   
    """
    # 원본 이미지와 Box 스케치 이미지 간 pixel 차이 감지
    diff_mask = np.sum(image != image_with_boxes, axis=-1) > 0

    # 차이가 없다면 None 반환 (스케치하지 않을 시)
    if np.max(diff_mask) == 0:
        return None
    
    # 차이가 존재하는 영역의 pixel의 색상 추출  
    box_colors = image_with_boxes[diff_mask]

    # 가장 빈도가 높은 색상을 box_color으로 간주    
    box_color = mode(box_colors, axis=0, keepdims=True)[0][0]

    # 전체 이미지에서 box_color와 일치하는 pixel만 골라내기
    color_mask = np.sum(image_with_boxes == box_color, axis=-1) == 3

    # 색상 일치(color_mask)와 pixel 차이(diff_mask)를 모두 만족하는 영역을 이진 처리
    bounding_mask = ((color_mask & diff_mask)).astype(np.uint8)

    # 이진 처리된 마스크로부터 최소 사이즈의 bounding box 좌표 계산   
    x, y, w, h = cv2.boundingRect(bounding_mask)
    box = np.array([x, y, x + w, y + h])
    return box

# 가이드 예측 함수   
@torch.no_grad()
def guided_prediction(
    predictor: SamPredictor,
    low_res_mask: np.ndarray,
    image: np.ndarray,
    fg_canvas: np.ndarray,
    bg_canvas: np.ndarray,
    box_canvas: np.ndarray,
    progress: gr.Progress = gr.Progress(),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    user가 스케치한 keypoint/box로부터 좌표와 라벨 생성,
    predictor.predict 호출하여 마스크 재예측    
    return:
        color_masks : 컬러로 시각화된 mask list
        low_res_mask : 다음 예측을 위한 Low-resolution mask    
    """
    # Foreground keypoint 찾기 시작 (0%)
    progress(0, "Finding foreground keypoints", total=4)
    fg_points = find_dots(image, fg_canvas)

    # Background keypoint 찾기 시작 (25%)
    progress(0.25, "Finding background keypoints", total=4)
    bg_points = find_dots(image, bg_canvas)

    # Bounding Box 검출 시작 (50%)
    progress(0.5, "Detecting bounding boxes", total=4)
    boxes = detect_boxes(image, box_canvas)

    # 좌표와 라벨 설정 (FG=1, BG=0)
    if fg_points.size == 0:
        # 둘 다 없으면 전체에 대해 자동 탐색
        if bg_points.size == 0:
            point_coords = None
            point_labels = None
        # FG 없으면 BG만 사용
        else:
            point_coords = bg_points
            point_labels = np.zeros(len(bg_points))
    # BG 없으면 FG만 사용
    elif bg_points.size == 0:
        point_coords = fg_points
        point_labels = np.ones(len(fg_points))
    # FG와 BG의 좌표와 라벨 결합   
    else:
        point_coords = np.concatenate([fg_points, bg_points])
        point_labels = np.concatenate([np.ones(len(fg_points)), np.zeros(len(bg_points))])

    # 마스크 예측 시작 (75%)
    progress(0.75, "Predicting masks", total=4)
    masks, scores, low_res_mask = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        mask_input=low_res_mask,
        box=boxes,
        multimask_output=True,  # 여러 개의 후보 마스크 생성
    )

    # 가장 높은 score(IOU)를 가진 마스크 선택
    low_res_mask = low_res_mask[[np.argmax(scores)]]
    masks = masks.astype(int)

    # 컬러 맵 마스크 생성 : 각각 R/G/B로 시각화   
    colors = [[(1, 0, 0)], [(0, 1, 0)], [(0, 0, 1)]]
    color_masks = [color.label2rgb(masks[i], image, c) for i, c in enumerate(colors)]
    
    # 전체 완료 (100%) + 결과 반환    
    progress(1, "Returning masks", total=4)

    return color_masks, low_res_mask


# Batch 예측 함수   
@torch.no_grad()
def batch_predict(
    predictor: SamPredictor,
    input_folder: str,
    dest_folder: str,
    mask_suffix: str,
    points_per_side: int,
    points_per_batch: int,
    pred_iou_thresh: float,
    stability_score_thresh: float,
    stability_score_offset: float,
    box_nms_thresh: float,
    crop_n_layers: int,
    crop_nms_thresh: float,
    crop_overlap_ratio: float,
    crop_n_points_downscale_factor: float,
    min_mask_region_area: float,
    progress: gr.Progress = gr.Progress(),
) -> List[np.ndarray]:
    """
    지정된 폴더 내 모든 이미지에 대해 SAM Automatic mask generator task 수행
    결과 저장 후 반환    
    """
    # 결과를 저장할 폴더를 지정할 경우, 디렉토리 생성
    if dest_folder is not None:
        Path(dest_folder).mkdir(exist_ok=True)
    
    # 입력한 폴더 내 모든 이미지 파일의 경로 수집 (.jpeg, .png, .jpg)
    image_files = [
        p.resolve() for p in Path(input_folder).rglob("**/*")
        if p.suffix in {".jpeg", ".png", ".jpg"}
    ]

    # SamAutomaticMaskGenerator 초기화   
    generator = SamAutomaticMaskGenerator(
        predictor.model,
        points_per_side,
        points_per_batch,
        pred_iou_thresh,
        stability_score_thresh,
        stability_score_offset,
        box_nms_thresh,
        crop_n_layers,
        crop_nms_thresh,
        crop_overlap_ratio,
        crop_n_points_downscale_factor,
        min_mask_region_area=min_mask_region_area,
    )

    # 컬러 마스크 결과를 저장할 리스트   
    masks = []
    # 각 이미지에 대해 반복 처리    
    for i, image_path in enumerate(image_files):
        # 진행 상황 업데이트  
        progress(i / len(image_files), desc=f"Predicting {str(image_path)}", total=len(image_files))
        
        # 이미지 읽기 후 BGR -> RGB로 변환  
        image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
        
        # Annotation mask 생성   
        annotations = generator.generate(image)

        # Color mask 시각화    
        color_mask = create_color_mask(image, annotations)

        # 만약 결과를 저장할 폴더가 지정되었다면 해당 파일에 작성   
        if dest_folder is not None:
            dest_file = Path(dest_folder) / f"{image_path.stem}_{mask_suffix or ''}{image_path.suffix}"
            cv2.imwrite(str(dest_file), color_mask * 255)
            
        masks.append(color_mask)

    return masks

# 스케치용 canvas 초기화 함수   
def set_sketch_images(image: np.ndarray) -> Tuple[np.ndarray, 
                                                  np.ndarray, 
                                                  np.ndarray]:
    """
    Predictor 탭에서 스케치용 canvas를 초기화할 때
    원본 이미지를 복사하여 반환
    """
    return image, image, image

# Image embedding 초기화 함수   
def compute_image_embedding(predictor: SamPredictor, image: np.ndarray):
    """
    predictor에 원본 이미지를 세팅하여 내부 임베딩을 준비    
    """
    predictor.set_image(image)
    return None


#---------------------모델 로딩 및 Gradio Interface 구성---------------------
# 모델 파일 리스트 호출
available_models = [x for x in os.listdir("models") if x.endswith(".pth")]

# 기본 모델 선택    
default_model = available_models[0]
device = "cuda" if torch.cuda.is_available() else "cpu"

# Default Predictor load    
default_predictor = load_model(default_model, device)

with gr.Blocks() as application:
    # ==== 앱 상단 헤더 ====
    # ==== Markdown 형식으로 제목 표시 ====   
    gr.Markdown(value="# Segment Anything FP16 Web Demo")
    selected_model = gr.Dropdown(choices=available_models, label="Model",
                                 value=default_model, interactive=True)

    # ==== 모델 선택 Dropdown UI ====  
    predictor_state = gr.State(default_predictor)
    # Dropdown 변경 시, 콜백 함수 실행 -> 새로운 모델 로드    
    selected_model.change(lambda x: load_model(x, device), inputs=[
                          selected_model], outputs=[predictor_state], show_progress=True)
    
    # ==== Automatic Segmentor 탭  ====
    with gr.Tab("Automatic Segmentor"):
        with gr.Row():
            # ==== 왼쪽 컬럼 ====
            # ==== 파라미터 설정 ====   
            with gr.Column():
                # Discrete settings Accordion pannel
                with gr.Accordion("Discrete Settings"):
                    with gr.Row():
                        # 각 파라미터 Number component로 설정   
                        points_per_side = gr.Number(label="points_per_side", value=32, precision=0)
                        points_per_batch = gr.Number(label="points_per_batch", value=64, precision=0)
                        stability_score_offset = gr.Number(label="stability_score_offset", value=1)
                        crop_n_layers = gr.Number(label="crop_n_layers", precision=0)
                        crop_n_points_downscale_factor = gr.Number(
                            label="crop_n_points_downscale_factor", value=1, precision=0)
                        min_mask_region_area = gr.Number(label="min_mask_region_area", precision=0, value=0)

                # Threshold settings Accordion pannel   
                with gr.Accordion("Threshold Settings"):
                    pred_iou_thresh = gr.Slider(label="pred_iou_thresh", minimum=0, maximum=1, value=0.88, step=0.01)
                    stability_score_thresh = gr.Slider(label="stability_score_thresh",
                                                       minimum=0, maximum=1, value=0.95, step=0.01)
                    box_nms_thresh = gr.Slider(label="box_nms_thresh", minimum=0, maximum=1, value=0.7)
                    crop_nms_thresh = gr.Slider(label="crop_nms_thresh", minimum=0, maximum=1, value=0.7)
                    crop_overlap_ratio = gr.Slider(label="crop_overlap_ratio", minimum=0,
                                                   maximum=1, value=512 / 1500, step=0.01)
            
            # ==== 오른쪽 컬럼 ====
            # ==== Single/Batch Prediction UI ==== 
            with gr.Column():
                # Single Prediction 탭   
                with gr.Tab("Single Prediction"):
                    # 이미지 업로드    
                    image = gr.Image(
                        source="upload",
                        label="Input Image",
                        elem_id="image",
                        brush_radius=20,
                    )
                    # Instance Segment 추출 옵션 pannel   
                    with gr.Accordion("Instance Segment Export Settings"):
                        display_rgba_segments = gr.Checkbox(label="Extract RGBA image for each mask")
                        mask_filter_area = gr.Number(label="Segment Mask Area Filter", precision=0, value=0)

                    # Predict 실행 버튼   
                    submit = gr.Button("Single Prediction")
                    # segmentation map 표시   
                    output = gr.Image(interactive=False, label="Segmentation Map")
                    # segment 별 이미지 갤러리 
                    annotation_masks = gr.Gallery(label="Segment Images")
                    # 버튼 클릭 시, generate 함수 호출   
                    submit.click(generate, inputs=[
                        predictor_state,
                        image,
                        points_per_side,
                        points_per_batch,
                        pred_iou_thresh,
                        stability_score_thresh,
                        stability_score_offset,
                        box_nms_thresh,
                        crop_n_layers,
                        crop_nms_thresh,
                        crop_overlap_ratio,
                        crop_n_points_downscale_factor,
                        min_mask_region_area,
                        display_rgba_segments,
                        mask_filter_area,
                    ], outputs=[output, annotation_masks])

                # Batch Prediction 탭 
                with gr.Tab("Batch Prediction"):
                    # 입력 폴더 경로    
                    input_folder = gr.Textbox(label="Image Folder")
                    # 출력 폴더 경로
                    dest_folder = gr.Textbox(label="Output Folder")
                    mask_suffix = gr.Textbox(label="Mask Suffix", value="seg")
                    batch_predict_button = gr.Button(value="Batch Predict")
                    # 출력 결과 이미지 갤러리
                    batch_outputs = gr.Gallery(label="Batch Outputs")
                    # 버튼 클릭 시 batch_predict 함수 호출
                    batch_predict_button.click(batch_predict, inputs=[
                        predictor_state,
                        input_folder,
                        dest_folder,
                        mask_suffix,
                        points_per_side,
                        points_per_batch,
                        pred_iou_thresh,
                        stability_score_thresh,
                        stability_score_offset,
                        box_nms_thresh,
                        crop_n_layers,
                        crop_nms_thresh,
                        crop_overlap_ratio,
                        crop_n_points_downscale_factor,
                        min_mask_region_area,
                    ], outputs=[batch_outputs])

application.queue()
application.launch()
