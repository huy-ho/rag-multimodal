from PIL import Image
import pytesseract
from sklearn.cluster import DBSCAN
import numpy as np

def extract_text_boxes(image_path):
    img = Image.open(image_path)
    ocr_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    
    boxes = []
    n_boxes = len(ocr_data['level'])
    for i in range(n_boxes):
        text = ocr_data['text'][i].strip()
        if text:
            x, y, w, h = ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i]
            cx, cy = x + w / 2, y + h / 2
            boxes.append({
                'text': text,
                'bbox': (x, y, w, h),
                'center': (cx, cy)
            })
    return boxes

def cluster_text_boxes(boxes, eps=50):
    centers = np.array([box['center'] for box in boxes])
    clustering = DBSCAN(eps=eps, min_samples=1).fit(centers)
    
    clusters = {}
    for idx, label in enumerate(clustering.labels_):
        clusters.setdefault(label, []).append(boxes[idx]['text'])
    
    return clusters

def main():
    image_path = r"C:\Users\huyho\OneDrive\Desktop\stuff\p-projects\rag-multimodal\data\diagrams\architectural-diagram-3.png"  # Adjust for your environment
    
    print(f"Loading image and extracting text from {image_path}...\n")
    boxes = extract_text_boxes(image_path)
    
    print(f"Extracted {len(boxes)} words from OCR.\n")
    
    print("Clustering words into components...")
    clusters = cluster_text_boxes(boxes, eps=50)
    
    print(f"\nDetected components (grouped text):")
    for cid, words in clusters.items():
        component_label = " ".join(words)
        print(f"- Component {cid}: {component_label}")

if __name__ == "__main__":
    main()
