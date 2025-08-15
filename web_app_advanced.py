import streamlit as st
import numpy as np
import pandas as pd
import os
import json
import re
from transformers import CLIPProcessor, CLIPModel
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import google.generativeai as genai
from ultralytics import YOLO
import math

# --- Streamlit UI ---
st.set_page_config(page_title="Advanced Scene Localization", layout="wide")
st.title("🔍 Advanced Scene Localization in Dense Images via Natural Language Queries")

# Sidebar for Gemini API key and settings
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Gemini API Key", type="password", help="Enter your Gemini API key")
    confidence_threshold = st.slider("Confidence Threshold", 0.1, 0.9, 0.3, 0.05)
    scale_factor = st.slider("Scale Factor", 0.5, 3.0, 1.1, 0.05)

# File uploader and query input
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])
query = st.text_area("Describe the scene or objects you want to find", height=80)

# Load models (cache for performance)
@st.cache_resource(show_spinner=True)
def load_models():
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    text_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    yolo_model = YOLO("yolov8x-oiv7.pt")
    return clip_model, text_processor, yolo_model


# --- All advanced functions from notebook ---
def get_yolo_detections(yolo_model, image_path):
    results = yolo_model.predict(source=image_path, verbose=False)
    df = results[0].to_df()
    if df.empty:
        return df
    cols = list(df.columns)
    name_count = 0
    new_cols = []
    for col in cols:
        if col == 'name':
            name_count += 1
            new_cols.append('class_id' if name_count > 1 else 'name')
        else:
            new_cols.append(col)
    df.columns = new_cols
    try:
        coords = df['box'].apply(lambda b: pd.Series([b['x1'], b['y1'], b['x2'], b['y2']]))
        coords.columns = ['xmin', 'ymin', 'xmax', 'ymax']
        df = pd.concat([df.drop('box', axis=1), coords], axis=1)
    except Exception as e:
        print(f"An error occurred while unpacking the 'box' column: {e}")
        return df.drop('box', axis=1, errors='ignore')
    return df

def plot_initial_boxes(image_path, df, color=(0, 128, 255), thickness=2):
    image_pil = Image.open(image_path)
    image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for _, row in df.iterrows():
        try:
            x_min, y_min, x_max, y_max = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            label = f"{row['name']} ({row['confidence']:.2f})"
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, thickness)
            cv2.rectangle(image, (x_min, y_min - text_height - 10), (x_min + text_width, y_min), color, -1)
            cv2.putText(image, label, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), thickness)
        except KeyError as e:
            continue
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)
    ax.axis("off")
    return fig

def clean_and_parse_json(text):
    cleaned = re.sub(r"```json\s*([\s\S]*?)\s*```", r"\1", text, flags=re.IGNORECASE)
    cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        print(f"JSON parsing failed after cleanup: {e}")
        return None

def refine_and_extract(query, genai_model):
    prompt = f"""
    You are an assistant helping to prepare queries for a vision-language model.
    Please do the following for the user query:
    1. Rewrite the query to be simple, adding all likely concerned parties involved, without changing the original meaning.
    2. List the main OBJECTS (nouns or tangible items) mentioned or implied.
      - Remove any duplicates or overlapping objects in the same category.
      - If multiple words refer to the same object type (e.g., 'man' and 'person'), include only one representative term.
      - Ignore the inanimate objects that do not explicitly help in the description of the scene.
    3. List the main ACTIONS (verbs) mentioned.
      - If multiple words refer to the same action (e.g., 'snatching' and 'pulling'), include only one representative term.
      - Do not include the verbs that are a weak representation of the actions in the prompt.
    Format the output exactly as JSON:
    {{
      "refined_prompt": "...",
      "objects": ["object1", "object2", ...],
      "actions": ["action1", "action2", ...]
    }}
    User query: "{query}"
    """
    response = genai_model.generate_content(prompt)
    text = response.text.strip()
    data = clean_and_parse_json(text)
    return data

def get_primary_object(object_list, genai_model, query):
    if not object_list:
        return None
    objects_str = ", ".join(object_list)
    prompt = f"""
    From the following list of objects found in a scene, identify the single 'primary_object'.
    The primary object should be the main anchor object that is involved in the actions taking place. This is usually a person, animal, or large vehicle/furniture.
    You can judge the primary_object using the prompt.
    Return ONLY the name of the single best primary object from the list, and nothing else.
    Object list: [{objects_str}]
    Prompt: {query}
    Primary Object:
    """
    try:
        response = genai_model.generate_content(prompt)
        primary_object = response.text.strip()
        return primary_object
    except Exception as e:
        print(f"Error while getting primary object: {e}")
        return None

def normalize_objects_with_nlm_prompt(prompt_refine_result, yolo_classes, gemini_model):
    if not prompt_refine_result:
        return None
    yolo_classes_str = ", ".join(yolo_classes.values())
    prompt = f"""
    You are a highly efficient and precise JSON generation assistant.
    Your ONLY task is to normalize a list of object names.
    Given a list of detected objects: {prompt_refine_result.get('objects', [])}
    And a list of valid YOLO classes: [{yolo_classes_str}]
    Map any synonyms or related terms from the detected objects to their canonical YOLO class names. For instance, map "man," "woman," "kid," or "child" to "person."
    Your output MUST be a valid JSON object. Do not include any conversational text, explanations, or markdown code block syntax (e.g., ```json).
    The JSON structure must be exactly as follows:
    {{
      "refined_prompt": "{prompt_refine_result.get('refined_prompt', '')}",
      "objects": ["object1", "object2", ...],
      "primary_object": "object_name",
      "actions": {prompt_refine_result.get('actions', '""')}
    }}
    """
    try:
        response = gemini_model.generate_content(prompt)
        text = response.text.strip()
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            cleaned_text = match.group(0)
            llm_result = json.loads(cleaned_text)
            prompt_refine_result['objects'] = llm_result.get('objects', prompt_refine_result['objects'])
            if 'primary_object' in llm_result:
                prompt_refine_result['primary_object'] = llm_result['primary_object'].lower()
            else:
                primary_obj_llm = llm_result.get('objects', [None])[0]
                prompt_refine_result['primary_object'] = primary_obj_llm.lower() if primary_obj_llm else None
            return prompt_refine_result
    except Exception as e:
        print(f"Failed to parse JSON from NLM normalization step: {e}")
    mapping = {'man': 'person', 'woman': 'person'}
    if 'primary_object' in prompt_refine_result and prompt_refine_result['primary_object']:
        primary_obj = prompt_refine_result['primary_object'].lower()
        prompt_refine_result['primary_object'] = mapping.get(primary_obj, primary_obj)
    if 'objects' in prompt_refine_result:
        normalized_objects = []
        for obj in prompt_refine_result['objects']:
            obj_lower = obj.lower()
            normalized_objects.append(mapping.get(obj_lower, obj_lower))
        prompt_refine_result['objects'] = sorted(list(set(normalized_objects)))
    return prompt_refine_result

def normalize_man_woman_person_to_person(detections_df):
    mapping = {'man': 'person', 'woman': 'person', 'person': 'person'}
    detections_df['name'] = detections_df['name'].apply(lambda x: mapping.get(x.lower(), x.lower()))
    return detections_df

def normalize_objects_with_nlm_dataframe(detections_df, yolo_classes, gemini_model):
    if detections_df.empty:
        return detections_df
    unique_detected_names = list(detections_df['name'].str.lower().unique())
    yolo_classes_str = ", ".join(yolo_classes.values())
    prompt = f"""
    You are an AI assistant specialized in normalizing object detection data.
    Your task is to create a mapping from detected object names to a canonical YOLO class name.
    Given a list of detected objects: {unique_detected_names}
    And a list of valid YOLO classes: [{yolo_classes_str}]
    Create a JSON object that maps each detected object name to its best-matching YOLO class name.
    If a detected object name is already a valid YOLO class, map it to itself.
    If an object does not have a clear canonical match, map it to itself.
    For example, if "man", "woman", and "child" are detected and "person" is a YOLO class,
    your mapping should be: {{"man": "person", "woman": "person", "child": "person"}}.
    Your output MUST be a valid JSON object containing only the mapping dictionary.
    Do not include any conversational text, explanations, or markdown code block syntax.
    Example output structure:
    {{
      "man": "person",
      "woman": "person",
      "car": "car"
    }}
    """
    try:
        response = gemini_model.generate_content(prompt)
        text = response.text.strip()
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            mapping_str = match.group(0)
            llm_mapping = json.loads(mapping_str)
        else:
            llm_mapping = {}
    except Exception as e:
        llm_mapping = {}
    detections_df['name'] = detections_df['name'].str.lower().apply(
        lambda x: llm_mapping.get(x, x)
    )
    return detections_df

def filter_detections_by_prompt_objects(detections_df, prompt_objects):
    prompt_objects_lower = set(obj.lower() for obj in prompt_objects)
    filtered_df = detections_df[detections_df['name'].str.lower().isin(prompt_objects_lower)].reset_index(drop=True)
    return filtered_df

def find_person_centric_interactions(filtered_df, prompt_refine_result, scale_factor=1.5):
    df = filtered_df.copy()
    if 'xmin' not in df.columns:
        if 'box' in df.columns:
            try:
                coords = df['box'].apply(pd.Series)
                df[['xmin', 'ymin', 'xmax', 'ymax']] = coords[['x1', 'y1', 'x2', 'y2']]
            except Exception as e:
                return pd.DataFrame()
        else:
            return pd.DataFrame()
    primary_object_name = prompt_refine_result.get("primary_object")
    if not primary_object_name:
        return pd.DataFrame()
    df['width'] = df['xmax'] - df['xmin']
    df['height'] = df['ymax'] - df['ymin']
    df['center_x'] = df['xmin'] + df['width'] / 2
    df['center_y'] = df['ymin'] + df['height'] / 2
    primary_df = df[df['name'].str.lower() == primary_object_name.lower()]
    secondary_df = df[df['name'].str.lower() != primary_object_name.lower()]
    if primary_df.empty or secondary_df.empty:
        return pd.DataFrame()
    interaction_boxes = []
    secondary_centers = secondary_df[['center_x', 'center_y']].to_numpy()
    for i, primary_obj in primary_df.iterrows():
        primary_center = np.array([primary_obj['center_x'], primary_obj['center_y']])
        primary_avg_size = (primary_obj['width'] + primary_obj['height']) / 2
        dynamic_threshold = primary_avg_size * scale_factor
        if len(secondary_centers) > 0:
            distances = np.linalg.norm(secondary_centers - primary_center, axis=1)
            nearby_indices = np.where(distances <= dynamic_threshold)[0]
            for idx in nearby_indices:
                secondary_obj = secondary_df.iloc[idx]
                box_pair = pd.concat([pd.DataFrame([primary_obj]), pd.DataFrame([secondary_obj])])
                xmin, ymin = box_pair[['xmin', 'ymin']].min()
                xmax, ymax = box_pair[['xmax', 'ymax']].max()
                interaction_boxes.append({
                    'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax,
                    'confidence': box_pair['confidence'].max(),
                    'objects': tuple(sorted(box_pair['name'].str.lower().tolist()))
                })
    if not interaction_boxes:
        return pd.DataFrame()
    return pd.DataFrame(interaction_boxes).drop_duplicates()

def plot_bounding_boxes(image_path, df, color=(255, 0, 0), thickness=2):
    image = cv2.imread(image_path)
    if image is None:
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for _, row in df.iterrows():
        x_min, y_min, x_max, y_max = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        label = ", ".join(row['objects']) if 'objects' in row else row.get('name', '')
        if 'confidence' in df.columns:
            label += f" ({row['confidence']:.2f})"
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, thickness)
        cv2.rectangle(image, (x_min, y_min - text_height - 10), (x_min + text_width, y_min), color, -1)
        cv2.putText(image, label, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), thickness)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)
    ax.axis("off")
    return fig

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    intersection_area = max(0, xB - xA) * max(0, yB - yA)
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union_area = float(boxA_area + boxB_area - intersection_area)
    iou = intersection_area / union_area if union_area > 0 else 0
    return iou

def merge_boxes_by_iou(df, iou_threshold=0.6):
    df_merged = df.copy()
    while True:
        merged_in_pass = False
        records = df_merged.to_dict('records')
        new_records = []
        merged_indices = set()
        for i in range(len(records)):
            if i in merged_indices:
                continue
            current_box = records[i]
            boxA_coords = [current_box['xmin'], current_box['ymin'], current_box['xmax'], current_box['ymax']]
            cluster_to_merge = [current_box]
            cluster_indices = {i}
            for j in range(i + 1, len(records)):
                if j in merged_indices:
                    continue
                other_box = records[j]
                boxB_coords = [other_box['xmin'], other_box['ymin'], other_box['xmax'], other_box['ymax']]
                if calculate_iou(boxA_coords, boxB_coords) > iou_threshold:
                    cluster_to_merge.append(other_box)
                    cluster_indices.add(j)
                    merged_in_pass = True
            if len(cluster_to_merge) > 1:
                all_xmins = [rec['xmin'] for rec in cluster_to_merge]
                all_ymins = [rec['ymin'] for rec in cluster_to_merge]
                all_xmaxs = [rec['xmax'] for rec in cluster_to_merge]
                all_ymaxs = [rec['ymax'] for rec in cluster_to_merge]
                union_box = {
                    'xmin': min(all_xmins),
                    'ymin': min(all_ymins),
                    'xmax': max(all_xmaxs),
                    'ymax': max(all_ymaxs),
                    'confidence': max([rec['confidence'] for rec in cluster_to_merge]),
                    'objects': tuple(sorted(list(set(obj for rec in cluster_to_merge for obj in rec['objects']))))
                }
                new_records.append(union_box)
                merged_indices.update(cluster_indices)
            else:
                new_records.append(current_box)
                merged_indices.add(i)
        df_merged = pd.DataFrame(new_records)
        if not merged_in_pass:
            break
    return df_merged

def crop_and_store_images(image_path, bounding_box_df):
    try:
        source_image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        return None
    cropped_images_list = []
    for _, row in bounding_box_df.iterrows():
        crop_box = (
            int(row['xmin']),
            int(row['ymin']),
            int(row['xmax']),
            int(row['ymax'])
        )
        cropped_img = source_image.crop(crop_box)
        cropped_images_list.append(cropped_img)
    df_with_crops = bounding_box_df.copy()
    df_with_crops['cropped_image'] = cropped_images_list
    return df_with_crops

def display_cropped_scenes(df_with_crops):
    if df_with_crops is None or 'cropped_image' not in df_with_crops.columns:
        return
    if df_with_crops.empty:
        return
    num_images = len(df_with_crops)
    num_cols = min(num_images, 3)
    num_rows = math.ceil(num_images / num_cols)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 5, num_rows * 5))
    if num_images > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    for i, (_, row) in enumerate(df_with_crops.iterrows()):
        ax = axes[i]
        img = row['cropped_image']
        label = ", ".join(row['objects']) if 'objects' in row else ''
        ax.imshow(img)
        ax.set_title(f"Scene {i+1}: {label}", fontsize=12)
        ax.axis('off')
    for j in range(num_images, len(axes)):
        axes[j].axis('off')
    plt.tight_layout(pad=2.0)
    st.pyplot(fig)

def calculate_clip_similarity(refined_prompt, df_with_crops, clip_model, processor):
    if df_with_crops is None or 'cropped_image' not in df_with_crops.columns:
        return None
    with torch.no_grad():
        text_inputs = processor(text=refined_prompt, return_tensors="pt", padding=True)
        text_features = clip_model.get_text_features(**text_inputs)
        text_features = F.normalize(text_features, p=2, dim=-1)
        image_list = df_with_crops['cropped_image'].tolist()
        image_inputs = processor(images=image_list, return_tensors="pt", padding=True)
        image_features = clip_model.get_image_features(**image_inputs)
        image_features = F.normalize(image_features, p=2, dim=-1)
        similarity_scores = (text_features @ image_features.T).squeeze(0)
    result_df = df_with_crops.copy()
    result_df['similarity_score'] = similarity_scores.cpu().numpy()
    return result_df

# Main logic
if uploaded_file and query and api_key:
    # Setup Gemini
    try:
        genai.configure(api_key=api_key)
        gemini_model = genai.GenerativeModel("gemini-2.5-pro")
    except Exception as e:
        st.error(f"Failed to configure Gemini: {e}")
        st.stop()

    # Load models
    clip_model, text_processor, model = load_models()

    # Save uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    image_path = "temp_uploaded_image.png"
    image.save(image_path)

    # --- Advanced pipeline logic (same as notebook, but with Streamlit display) ---
    bounding_boxes_df = get_yolo_detections(model, image_path)
    st.subheader("YOLO Detections")
    st.dataframe(bounding_boxes_df)
    st.pyplot(plot_initial_boxes(image_path, bounding_boxes_df))

    prompt_refine_result = refine_and_extract(query, gemini_model)
    if prompt_refine_result and 'objects' in prompt_refine_result:
        object_list = prompt_refine_result['objects']
        primary_object = get_primary_object(object_list, gemini_model, prompt_refine_result['refined_prompt'])
        prompt_refine_result['primary_object'] = primary_object
        st.write(f"Initial objects from prompt: {object_list}")
        st.write(f"Identified Primary Object: {primary_object}")

    all_yolo_classes = model.names
    yolo_mapped_result = normalize_objects_with_nlm_prompt(prompt_refine_result, all_yolo_classes, gemini_model)
    prompt_refine_result['objects'] = yolo_mapped_result['objects']

    results = model.predict(source=image_path, verbose=False)
    df = results[0].to_df()
    df = normalize_objects_with_nlm_dataframe(df, all_yolo_classes, gemini_model)
    st.subheader("Normalized Detections")
    st.dataframe(df)

    filtered_df = filter_detections_by_prompt_objects(df, prompt_refine_result["objects"])
    final_scene_df = find_person_centric_interactions(
        filtered_df=filtered_df,
        prompt_refine_result=prompt_refine_result,
        scale_factor=scale_factor
    )
    st.subheader("Person-Centric Interactions")
    st.dataframe(final_scene_df)
    if not final_scene_df.empty:
        st.pyplot(plot_bounding_boxes(image_path, final_scene_df))
    else:
        st.warning("No interaction boxes were created.")

    df_cropped_bounding_boxes = merge_boxes_by_iou(final_scene_df, iou_threshold=0.35)
    st.subheader("Merged Boxes by IoU")
    st.dataframe(df_cropped_bounding_boxes)
    if not df_cropped_bounding_boxes.empty:
        st.pyplot(plot_bounding_boxes(image_path, df_cropped_bounding_boxes, color=(0, 255, 0)))
    else:
        st.warning("No boxes remained after merging.")

    scenes_with_images_df = crop_and_store_images(image_path, df_cropped_bounding_boxes)
    if scenes_with_images_df is not None:
        st.success("Successfully created DataFrame with cropped images.")
        st.dataframe(scenes_with_images_df[['objects', 'cropped_image']])

    st.subheader("Cropped Scenes")
    display_cropped_scenes(scenes_with_images_df)
    refined_prompt_text = prompt_refine_result['refined_prompt']
    df_with_scores = calculate_clip_similarity(
        refined_prompt=refined_prompt_text,
        df_with_crops=scenes_with_images_df,
        clip_model=clip_model,
        processor=text_processor
    )
    if df_with_scores is not None:
        st.subheader("CLIP Similarity Scores")
        sorted_df = df_with_scores.sort_values(by='similarity_score', ascending=False)
        st.dataframe(sorted_df[['objects', 'similarity_score']])
        if not sorted_df.empty:
            top_3_scenes = sorted_df.head(3)
            st.subheader("Top 3 Matches")
            display_cropped_scenes(top_3_scenes)
else:
    st.info("Please upload an image, enter a query, and provide your Gemini API key.")
