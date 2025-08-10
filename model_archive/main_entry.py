from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from model_archive.model import DINOv2TokenSegmentation, YOLOv9_M4, MaskRCNNSwin, EMA, UNETR_MoE_CLIP_RCNN
from model_archive.dataset import SegmentationDataset, YoloDetectionDataset, SegmentationDataset_ema, MultiModalSegDataset, MultiModalSegCascadeDataset
from model_archive.train_val import (train_seg, evaluate_seg, test_seg, inference_seg, \
                       train_yolo, evaluate_yolo, test_yolo, inference_yolo, \
                       train_maskrcnn_ema, evaluate_maskrcnn_ema, inference_maskrcnn_ema, \
                       train_segmentation_model_moe, validate_segmentation_model_moe, evaluate_segmentation_model_moe, inference_segmentation_mode_moe)
from model_archive.loss import Mask2FormerLoss, CLIP_MultiTaskLoss
from model_archive.utils_func import collate_fn, collate_fn_yolo, collate_fn_moe, model_info, load_checkpoint, optimizer_setup
from model_archive.config import Config
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, MultiStepLR, ExponentialLR, ReduceLROnPlateau, OneCycleLR
from torch.utils.tensorboard import SummaryWriter
import os

def model_trainvaltest_process(optimizer_type="adam",
                               lr=1e-4,
                               scheduler_mode="cosineannel",
                               epochs=10,
                               mode="train", 
                               ml="dinov2",
                               model_tuning_enable=False, 
                               log_enable=False,
                               start_epoch=1,
                               input_inference_path=None,
                               save_dir=None,
                               progress_path=None,
                               patient_id=None,
                               db_path=None):

    all_config = Config()

    device = all_config.device
    class_color_map = all_config.class_color_map

    train_image_paths = all_config.train_image_paths
    train_ann_paths = all_config.train_ann_paths

    val_image_paths = all_config.val_image_paths
    val_ann_paths = all_config.val_ann_paths

    test_image_paths = all_config.test_image_paths
    test_ann_paths = all_config.test_ann_paths

    # inference_image_paths = all_config.inference_image_paths
    inference_image_paths=input_inference_path
    
    img_size = all_config.img_size
    model_dir = all_config.model_dir
    lr = all_config.lr
    weight_decay = all_config.weight_decay
    batch_size = all_config.batch_size
    num_classes = all_config.num_classes
    class_names = all_config.class_names

    transform = all_config.transform

    image_transform = all_config.image_transform
    mask_transform = all_config.mask_transform

    image_transform_ema = all_config.image_transform_ema
    mask_transform_ema = all_config.mask_transform_ema

    image_transform_moe = all_config.image_transform_moe

    image_transform_dinov2 = all_config.image_transform_dinov2
    mask_transform_dinov2 = all_config.mask_transform_dinov2

    image_transform_cascade = all_config.image_transform_cascade

    # mask_transform_moe = all_config.mask_transform_moe
    index_to_classes_dict = all_config.index_to_classes_dict
    resize_img_size = all_config.resize_img_size
    # save_dir = all_config.save_dir
    db_path = db_path

    if log_enable:
        log_dir = f"runs/{ml}"
        if os.path.exists(log_dir) and not os.path.isdir(log_dir):
            os.remove(log_dir)
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
    else:
        writer = None

    if ml == "dinov2":
        model = DINOv2TokenSegmentation(num_classes=num_classes).to(device)  # 假設 3 類
        model = model.to(device).half() if device.type == "cuda" else model.to(device)
        loss_fn = Mask2FormerLoss(num_classes=num_classes)

        if model_tuning_enable == True:
            # 選擇只訓練 prompt 與 decoder 等模組的參數
            trainable_parameters = list(model.prompt.parameters()) + \
                            list(model.query_embed.parameters()) + \
                            list(model.transformer_decoder.parameters()) + \
                            list(model.class_head.parameters()) + \
                            list(model.mask_embed_head.parameters())

            model_info(model, trainable_parameters)
        else:
            trainable_parameters = model.parameters()

        if mode == "train":
            train_dataset = SegmentationDataset(img_size, train_image_paths, train_ann_paths, image_transform=image_transform_dinov2, mask_transform=mask_transform_dinov2)
            val_dataset = SegmentationDataset(img_size, val_image_paths, val_ann_paths, image_transform=image_transform_dinov2, mask_transform=mask_transform_dinov2)

            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

            steps_per_epoch = len(train_dataloader)
            optimizer, scheduler = optimizer_setup(optimizer_type, scheduler_mode, trainable_parameters, steps_per_epoch, lr, weight_decay, epochs, train_dataloader)

            # Training loop
            for epoch in range(epochs):
                print(f"Epoch {epoch+1}")
                loss, training_status = train_seg(model, epoch, epochs, train_dataloader, optimizer, scheduler, loss_fn, device, progress_path)

                if training_status["cancel"] == True:
                    break
                
                evaluate_seg(model, val_dataloader, loss_fn, device)
                torch.save({
                    'epoch': epoch+1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                }, f"{model_dir}/dinov2_token_segmentation_epoch{epoch+1}.pth")

            torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, f"{model_dir}/dinov2_token_segmentation_final.pth")
            
        else:
            
            if mode == "continue_train":

                train_dataset = SegmentationDataset(img_size, train_image_paths, train_ann_paths, image_transform=image_transform_dinov2, mask_transform=mask_transform_dinov2)
                val_dataset = SegmentationDataset(img_size, val_image_paths, val_ann_paths, image_transform=image_transform_dinov2, mask_transform=mask_transform_dinov2)

                train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
                val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

                steps_per_epoch = len(train_dataloader)
                optimizer, scheduler = optimizer_setup(optimizer_type, scheduler_mode, trainable_parameters, steps_per_epoch, lr, weight_decay, epochs, train_dataloader)

                checkpoint = torch.load(f"{model_dir}/dinov2_token_segmentation_final.pth")
                model.load_state_dict(checkpoint['model_state_dict'])
                start_epoch = checkpoint['epoch']
                epochs = start_epoch + epochs
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])    

                for epoch in range(start_epoch+1, epochs):
                    print(f"Epoch {epoch+1}")
                    loss, training_status = train_seg(model, epoch, epochs, train_dataloader, optimizer, loss_fn, device, progress_path)

                    if training_status["cancel"] == True:
                        break

                    evaluate_seg(model, val_dataloader, loss_fn, device)
                    torch.save({
                        'epoch': epoch+1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,
                    }, f"{model_dir}/dinov2_token_segmentation_{epoch+1}.pth")

                torch.save({
                    'epoch': epochs,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                }, f"{model_dir}/dinov2_token_segmentation_final.pth")
                
            elif mode == "test":
                checkpoint = torch.load(f"{model_dir}/dinov2_token_segmentation_final.pth")
                model.load_state_dict(checkpoint['model_state_dict'])
                
                test_dataset = SegmentationDataset(img_size, test_image_paths, test_ann_paths, image_transform=image_transform_dinov2, mask_transform=mask_transform_dinov2)
                test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

                test_seg(model, test_loader, device, save_dir=save_dir)

            elif mode == "inference":
                checkpoint = torch.load(f"{model_dir}/dinov2_token_segmentation_final.pth")
                model.load_state_dict(checkpoint['model_state_dict'])

                inference_dataset = SegmentationDataset(img_size, inference_image_paths, None, image_transform=image_transform_dinov2, mask_transform=None)
                inference_loader = DataLoader(inference_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
                
                inference_seg(model, inference_loader, device, input_inference_path=input_inference_path, save_dir=save_dir, progress_path=progress_path, patient_id=patient_id, db_path=db_path)

    elif ml == "yolov9":

        model = YOLOv9_M4(num_classes=num_classes).to(device)

        if model_tuning_enable == True:
            # 選擇只訓練 prompt 與 decoder 等模組的參數
            trainable_parameters = list(model.prompt.parameters()) + \
                            list(model.query_embed.parameters()) + \
                            list(model.transformer_decoder.parameters()) + \
                            list(model.class_head.parameters()) + \
                            list(model.mask_embed_head.parameters())

            model_info(model, trainable_parameters)
        else:
            trainable_parameters = model.parameters()

        if mode == "train":
            # Training loop

            train_dataset = YoloDetectionDataset(train_image_paths, train_ann_paths, transform)
            val_dataset = YoloDetectionDataset(val_image_paths, val_ann_paths, transform)
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_yolo)
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_yolo)

            steps_per_epoch = len(train_dataloader)
            optimizer, scheduler = optimizer_setup(optimizer_type, scheduler_mode, trainable_parameters, steps_per_epoch, lr, weight_decay, epochs, train_dataloader)

            for epoch in range(epochs):
                print(f"Epoch {epoch+1}")
                loss, training_status = train_yolo(model, epoch, epochs, train_dataloader, optimizer, device, num_classes, progress_path)

                if training_status["cancel"] == True:
                    break

                metrics = evaluate_yolo(model, val_dataloader, device, num_classes)

                torch.save({
                        'epoch': epoch+1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()
                    }, f"{model_dir}/yolov9_m4_detection_{epoch+1}.pth")

            torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, f"{model_dir}/yolov9_m4_detection_final.pth")

        elif mode == "continue_train":
            
            train_dataset = YoloDetectionDataset(train_image_paths, train_ann_paths, transform)
            val_dataset = YoloDetectionDataset(val_image_paths, val_ann_paths, transform)
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_yolo)
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_yolo)

            steps_per_epoch = len(train_dataloader)
            optimizer, scheduler = optimizer_setup(optimizer_type, scheduler_mode, trainable_parameters, steps_per_epoch, lr, weight_decay, epochs, train_dataloader)

            checkpoint = torch.load(f"{model_dir}/yolov9_m4_detection_final.pth")
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            start_epoch = checkpoint['epoch']
            epochs = start_epoch + epochs

            for epoch in range(start_epoch+1, epochs):
                print(f"Epoch {epoch+1}")
                loss, training_status = train_yolo(model, epoch, epochs, train_dataloader, optimizer, device, num_classes, progress_path)

                if training_status["cancel"] == True:
                    break

                metrics = evaluate_yolo(model, val_dataloader, device, num_classes)

                torch.save({
                        'epoch': epoch+1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()
                    }, f"{model_dir}/yolov9_m4_detection_{epoch+1}.pth")
        
        elif mode == "test":

            checkpoint = torch.load(f"{model_dir}/yolov9_m4_detection_final.pth")
            model.load_state_dict(checkpoint['model_state_dict'])

            test_dataset = YoloDetectionDataset(test_image_paths, test_ann_paths, transform)
            test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn_yolo)

            test_yolo(model, test_dataloader, device, class_names, save_dir=save_dir)
        
        elif mode == "inference":

            checkpoint = torch.load(f"{model_dir}/yolov9_m4_detection_final.pth")
            model.load_state_dict(checkpoint['model_state_dict'])

            inference_dataset = YoloDetectionDataset(test_image_paths, test_ann_paths, transform)
            inference_dataloader = DataLoader(inference_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn_yolo)

            inference_yolo(model, inference_dataloader, device, class_names, save_dir=save_dir, patient_id=patient_id, db_path=db_path)

    elif ml == "mask2former":

        model = MaskRCNNSwin(num_classes=3).to(device)    
        ema = EMA(model)

        if model_tuning_enable == True:
            # 選擇只訓練 prompt 與 decoder 等模組的參數
            trainable_parameters = list(model.prompt.parameters()) + \
                            list(model.query_embed.parameters()) + \
                            list(model.transformer_decoder.parameters()) + \
                            list(model.class_head.parameters()) + \
                            list(model.mask_embed_head.parameters())

            model_info(model, trainable_parameters)
        else:
            trainable_parameters = model.parameters()
        
        if mode == "train":
            train_dataset = SegmentationDataset_ema(img_size, train_image_paths, train_ann_paths, image_transform=image_transform_ema, mask_transform=mask_transform_ema, box_transform=transform)
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
            val_dataset = SegmentationDataset_ema(img_size, val_image_paths, val_ann_paths, image_transform=image_transform_ema, mask_transform=mask_transform_ema, box_transform=transform)
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

            steps_per_epoch = len(train_dataloader)
            optimizer, scheduler = optimizer_setup(optimizer_type, scheduler_mode, trainable_parameters, steps_per_epoch, lr, weight_decay, epochs, train_dataloader)

            # Training loop
            for epoch in range(epochs):
                print(f"Epoch {epoch+1}")
                loss, training_status = train_maskrcnn_ema(model, epoch, epochs, ema, train_dataloader, optimizer, device, batch_size, num_classes, writer=writer)

                if training_status["cancel"] == True:
                    break

                ema_model = ema.ema_model
                val_loss = evaluate_maskrcnn_ema(ema_model, val_dataloader, device, epoch, num_classes, class_names, class_color_map, save_dir=save_dir, writer=writer)
                
                torch.save({
                        'epoch': epoch+1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': val_loss
                    }, f"{model_dir}/mask2former_segmentation_epoch{epoch+1}.pth")
            
            torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss
            }, f"{model_dir}/mask2former_segmentation_final.pth")
            
        elif mode == "continue_train":

            train_dataset = SegmentationDataset_ema(img_size, train_image_paths, train_ann_paths, image_transform=image_transform_ema, mask_transform=mask_transform_ema, box_transform=transform)
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
            val_dataset = SegmentationDataset_ema(img_size, val_image_paths, val_ann_paths, image_transform=image_transform_ema, mask_transform=mask_transform_ema, box_transform=transform)
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

            steps_per_epoch = len(train_dataloader)
            optimizer, scheduler = optimizer_setup(optimizer_type, scheduler_mode, trainable_parameters, steps_per_epoch, lr, weight_decay, epochs, train_dataloader)

            checkpoint = torch.load(f"{model_dir}/mask2former_segmentation_final.pth")
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            start_epoch = checkpoint['epoch']
            epochs = start_epoch + epochs

            for epoch in range(start_epoch+1, epochs):
                print(f"Epoch {epoch+1}")
                loss, training_status = train_maskrcnn_ema(model, epoch, epochs, ema, train_dataloader, optimizer, device, batch_size, num_classes, writer=writer, progress_path=progress_path)

                if training_status["cancel"] == True:
                    break

                ema_model = ema.ema_model
                val_loss = evaluate_maskrcnn_ema(ema_model, val_dataloader, device, epoch, num_classes, save_dir=save_dir, writer=writer)
                
                torch.save({
                        'epoch': epoch+1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': val_loss
                    }, f"{model_dir}/mask2former_segmentation_epoch{epoch+1}.pth")

        elif mode == "test":

            checkpoint = torch.load(f"{model_dir}/mask2former_segmentation_final.pth")
            model.load_state_dict(checkpoint['model_state_dict'])

            test_dataset = SegmentationDataset_ema(img_size, test_image_paths, test_ann_paths, image_transform=image_transform_ema, mask_transform=mask_transform_ema, box_transform=transform)
            test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
            
            inference_maskrcnn_ema(model, test_dataloader, device, class_names=class_names, class_color_map=class_color_map, visualize=False, save_dir=save_dir, conf_thresh=0.5, patient_id=patient_id, db_path=db_path)

    elif ml == "unetr_moe":

        model = UNETR_MoE_CLIP_RCNN(
            in_channels=num_classes,
            out_channels=num_classes+1,           # 分類數量
            img_size=resize_img_size,
            feature_size=16,
            hidden_size=768,
            num_layers=12,
            roi_output_size=(7,7),
            num_heads=12              # 自動算 hidden_size，會是 num_patches * base_dim 向上取整
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        '''
        if model_tuning_enable == True:
            # 選擇只訓練 prompt 與 decoder 等模組的參數
            trainable_parameters = list(model.prompt.parameters()) + \
                            list(model.query_embed.parameters()) + \
                            list(model.transformer_decoder.parameters()) + \
                            list(model.class_head.parameters()) + \
                            list(model.mask_embed_head.parameters())

            model_info(model, trainable_parameters)
        else:
            trainable_parameters = model.parameters()
        '''
        trainable_parameters = model.parameters()

        criterion = CLIP_MultiTaskLoss()

        if mode == "train":
            # Training loop
            # best_dice = 0.0

            #with mlflow.start_run() as run:
            #    run_id = run.info.run_id
            #    print("Run ID:", run_id)

            train_dataset = MultiModalSegDataset(img_size, train_image_paths, train_ann_paths, image_transform=image_transform_moe, resize_img_size=resize_img_size, label_map=index_to_classes_dict)
            val_dataset = MultiModalSegDataset(img_size, val_image_paths, val_ann_paths, image_transform=image_transform_moe, resize_img_size=resize_img_size, label_map=index_to_classes_dict)
            
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_moe)
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_moe)

            steps_per_epoch = len(train_dataloader)
            optimizer, scheduler = optimizer_setup(optimizer_type, scheduler_mode, trainable_parameters, steps_per_epoch, lr, weight_decay, epochs, train_dataloader)

            for epoch in range(epochs):
                train_loss, training_status = train_segmentation_model_moe(model, epoch, epochs, train_dataloader, optimizer, criterion, scheduler, device, writer=writer, progress_path=progress_path)
                if training_status["cancel"] == True:
                    break

                val_dict = validate_segmentation_model_moe(model, val_dataloader, criterion, scheduler, device, epoch, writer=writer)

                val_loss = val_dict["val_loss"]
                acc = val_dict["val_acc"]
                precision = val_dict["val_precision"]
                iou = val_dict["val_iou"]

                #mlflow.log_metric("accuracy", acc)
                #mlflow.log_metric("precision", precision)
                #mlflow.log_metric("iou", iou)
                #mlflow.log_metric("train_loss", train_loss)
                #mlflow.log_metric("val_loss", val_loss)
                #mlflow.sklearn.log_model(model, "unetr_moe_model")
                torch.save(model.state_dict(), f"{model_dir}/unetr_moe_model_epoch{epoch+1}.pth")

            # mlflow.register_model(model_uri=f"runs:/{run_id}/unetr_moe_model", name="UNETR_MOE_Model")
            torch.save(model.state_dict(), f"{model_dir}/unetr_moe_model_final.pth")

        elif mode == "continue_train":
            
            train_dataset = MultiModalSegDataset(img_size, train_image_paths, train_ann_paths, image_transform=image_transform_moe, resize_img_size=resize_img_size, label_map=index_to_classes_dict)
            val_dataset = MultiModalSegDataset(img_size, val_image_paths, val_ann_paths, image_transform=image_transform_moe, resize_img_size=resize_img_size, label_map=index_to_classes_dict)
            
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_moe)
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_moe)

            steps_per_epoch = len(train_dataloader)
            optimizer, scheduler = optimizer_setup(optimizer_type, scheduler_mode, trainable_parameters, steps_per_epoch, lr, weight_decay, epochs, train_dataloader)

            checkpoint = load_checkpoint(model, optimizer, f"{model_dir}/unetr_moe_model_final.pth", device)
            epochs = checkpoint['epoch'] + epochs

            #with mlflow.start_run() as run:
            #    run_id = run.info.run_id
            #    print("Run ID:", run_id)

            for epoch in range(start_epoch, epochs):
                train_loss, training_status = train_segmentation_model_moe(model, epoch, epochs, train_dataloader, optimizer, criterion, device, epoch, print_every=10, writer=writer, progress_path=progress_path)
                if training_status["cancel"] == True:
                    break
                val_dict = validate_segmentation_model_moe(model, val_dataloader, criterion, device, epoch, writer=writer)

                val_loss = val_dict["val_loss"]
                acc = val_dict["val_acc"]
                precision = val_dict["val_precision"]
                iou = val_dict["val_iou"]

                #mlflow.log_metric("accuracy", acc)
                #mlflow.log_metric("precision", precision)
                #mlflow.log_metric("iou", iou)
                #mlflow.log_metric("train_loss", train_loss)
                #mlflow.log_metric("val_loss", val_loss)
                #mlflow.sklearn.log_model(model, "model")
                torch.save(model.state_dict(), f"{model_dir}/unetr_moe_model_epoch{epoch+1}.pth")

                #mlflow.register_model(model_uri=f"runs:/{run_id}/unetr_moe_model", name="UNETR_MOE_Model")
            
            torch.save(model.state_dict(), f"{model_dir}/unetr_moe_model_final.pth")

        elif mode == "test":
            
            test_dataset = MultiModalSegDataset(img_size, test_image_paths, test_ann_paths, image_transform=image_transform_moe, resize_img_size=resize_img_size, label_map=index_to_classes_dict)
            test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn_moe)

            model.load_state_dict(torch.load(f"{model_dir}/unetr_moe_model_final.pth"))
            # model = mlflow.pytorch.load_model(f"runs:/{run_id}/unetr_moe_model")
            # checkpoint = load_checkpoint(model, optimizer, f"{model_dir}/unetr_moe_best_model_epoch{start_epoch}.pth", device)
            evaluate_segmentation_model_moe(model, test_dataloader, criterion, device, num_classes=num_classes+1, class_names=class_names, visualize_cm=False)
        
        elif mode == "inference":
            
            if inference_image_paths:
                inference_dataset = MultiModalSegDataset(img_size, inference_image_paths, ann_paths=None, image_transform=image_transform_moe, resize_img_size=resize_img_size, label_map=index_to_classes_dict)
                inference_dataloader = DataLoader(inference_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn_moe)
            else:
                exit(-1)

            model.load_state_dict(torch.load(f"{model_dir}/unetr_moe_model_final.pth"))
            # model = mlflow.pytorch.load_model(f"runs:/{run_id}/unetr_moe_model")
            # checkpoint = load_checkpoint(model, optimizer, f"{model_dir}/unetr_moe_best_model_epoch{start_epoch}.pth", device)
            inference_segmentation_mode_moe(model, inference_dataloader, device, class_color_map, class_names=class_names, save_dir=save_dir, patient_id=patient_id, db_path=db_path)

    elif ml == "cascade_resnet":

        model = CascadeRCNN(img_size=img_size).to(device)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model_tuning_enable == True:
            # 選擇只訓練 prompt 與 decoder 等模組的參數
            trainable_parameters = list(model.prompt.parameters()) + \
                            list(model.query_embed.parameters()) + \
                            list(model.transformer_decoder.parameters()) + \
                            list(model.class_head.parameters()) + \
                            list(model.mask_embed_head.parameters())

            model_info(model, trainable_parameters)
        else:
            trainable_parameters = model.parameters()

        if mode == "train":
            
            train_dataset = MultiModalSegCascadeDataset(img_size, train_image_paths, train_ann_paths, image_transform=image_transform_cascade, resize_img_size=resize_img_size, label_map=index_to_classes_dict)
            val_dataset = MultiModalSegCascadeDataset(img_size, val_image_paths, val_ann_paths, image_transform=image_transform_cascade, resize_img_size=resize_img_size, label_map=index_to_classes_dict)
            
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_cascade)
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_cascade)

            steps_per_epoch = len(train_dataloader)
            optimizer, scheduler = optimizer_setup(optimizer_type, scheduler_mode, trainable_parameters, steps_per_epoch, lr, weight_decay, epochs, train_dataloader)

            # Training Loop
            for epoch in range(epochs):
                print(f"Epoch {epoch+1}")

                train_cascade_resnet(model, train_dataloader, device, epochs, optimizer, scheduler)

                evaluate_cascade_resnet(model, val_dataloader, device)

                # print(f"Epoch {epoch} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

                torch.save(model.state_dict(), f"{model_dir}/cascade_resnet_model_epoch{epoch+1}.pth")

            torch.save(model.state_dict(), f"{model_dir}/cascade_resnet_model_final.pth")
        
        elif mode == "continue_train":
            
            train_dataset = MultiModalSegCascadeDataset(img_size, train_image_paths, train_ann_paths, image_transform=image_transform_cascade, resize_img_size=resize_img_size, label_map=index_to_classes_dict)
            val_dataset = MultiModalSegCascadeDataset(img_size, val_image_paths, val_ann_paths, image_transform=image_transform_cascade, resize_img_size=resize_img_size, label_map=index_to_classes_dict)
            
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_cascade)
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_cascade)

            steps_per_epoch = len(train_dataloader)
            optimizer, scheduler = optimizer_setup(optimizer_type, scheduler_mode, trainable_parameters, steps_per_epoch, lr, weight_decay, epochs, train_dataloader)

            checkpoint = load_checkpoint(model, optimizer, f"{model_dir}/cascade_resnet_model_final.pth", device)
            epochs = checkpoint['epoch'] + epochs

            #with mlflow.start_run() as run:
            #    run_id = run.info.run_id
            #    print("Run ID:", run_id)

            for epoch in range(start_epoch, epochs):
                print(f"Epoch {epoch+1}")

                train_cascade_resnet(model, train_dataloader, device, epochs, optimizer, scheduler)

                evaluate_cascade_resnet(model, val_dataloader, device)

                print(f"Epoch {epoch} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

                torch.save(model.state_dict(), f"{model_dir}/cascade_resnet_model_epoch{epoch+1}.pth")
            
            torch.save(model.state_dict(), f"{model_dir}/cascade_resnet_model_final.pth")
        
        elif mode == "test":

            test_dataset = MultiModalSegCascadeDataset(img_size, val_image_paths, val_ann_paths, image_transform=image_transform_cascade, resize_img_size=resize_img_size, label_map=index_to_classes_dict)
            test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn_cascade)

            optimizer, _ = optimizer_setup(optimizer_type, scheduler_mode, trainable_parameters, steps_per_epoch, lr, weight_decay, epochs, train_dataloader)

            model.load_state_dict(torch.load(f"{model_dir}/cascade_resnet_model_final.pth"))
            # checkpoint = load_checkpoint(model, optimizer, f"{model_dir}/cascade_resnet_model_final.pth", device)

            test_cascade_resnet(model, test_dataloader, device, num_classes, class_names)

        elif mode == "inference":
            
            inference_dataset = MultiModalSegCascadeDataset(img_size, val_image_paths, val_ann_paths, image_transform=image_transform_cascade, resize_img_size=resize_img_size, label_map=index_to_classes_dict)
            inference_dataloader = DataLoader(inference_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn_cascade)

            inference_cascade_resnet(model, inference_dataloader, class_names, device)

if __name__ == "__main__":

    '''
    optimizer_type = session.get("optimizer_type", "adam")
    lr = session.get("lr", 1e-4)
    scheduler_mode = session.get("scheduler_mode", "cosineanneal")
    epochs = int(session.get("total_epochs", 10))
    ml = session.get("ml", "dinov2")
    model_tuning_enable = session.get("model_tuning_enable", False)
    log_enable = session.get("tensorboard_enable", False)
    start_epoch = int(session.get("start_epoch", 0))
    input_inference_path = sorted(glob.glob(f"{UPLOAD_DIR}/{patient_id}/*.png"))
    save_dir = f"{RESULT_DIR}/{patient_id}"
    '''

    model_trainvaltest_process(
        optimizer_type="adam",
        lr=1e-4,
        scheduler_mode="cosineanneal",
        epochs=10,
        mode="train",
        ml="cascade_resnet",
        model_tuning_enable=False,
        log_enable=False,
        start_epoch=1,
        input_inference_path=None,
        save_dir=None,
        progress_path=None,
        patient_id=None,
        db_path=None
    )