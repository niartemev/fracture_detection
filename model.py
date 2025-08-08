#Model training and metrics evaluation

#import libraries
from ultralytics import YOLO


def main():
    
    #Load pretrained model
    model = YOLO('yolo11n.pt')


    #Train using optimized parameters
    results = model.train(
    data = os.path.join(current_dir, "data.yaml"),
    epochs = 34,
    batch=8,
    imgsz = 640,
    lr0=0.01,
    name='bone_fracture_detection',
    project='/runs', device=1)
    
    #Load best model
    trained_model = YOLO("runs/bone_fracture_detection/weights/best.pt")
    
    #Evaluate metrics
    metrics = trained_model.val(conf=0.25)

    precision = metrics.results_dict['metrics/precision(B)']
    recall = metrics.results_dict['metrics/recall(B)']

    F1 = 2 * (precision * recall) / (precision + recall)
    
    #Print metrics
    print("\nValidation Metrics:")
    print(f"mAP50: {metrics.results_dict['metrics/mAP50(B)']:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 score: {F1:.4f}")

    

main()
