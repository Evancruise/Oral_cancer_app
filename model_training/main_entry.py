from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from model_training.model import DINOv2TokenSegmentation, YOLOv9_M4, MaskRCNNSwin, EMA, UNETR_MoE_CLIP_RCNN
from model_training.dataset import SegmentationDataset, YoloDetectionDataset, SegmentationDataset_ema, MultiModalSegDataset
from model_training.train_val import (train_seg, evaluate_seg, test_seg, inference_seg, \
                       train_yolo, evaluate_yolo, test_yolo, inference_yolo, \
                       train_maskrcnn_ema, evaluate_maskrcnn_ema, inference_maskrcnn_ema, \
                       train_segmentation_model_moe, validate_segmentation_model_moe, evaluate_segmentation_model_moe, inference_segmentation_mode_moe)
from model_training.loss import Mask2FormerLoss, CLIP_MultiTaskLoss
from model_training.utils import collate_fn, collate_fn_yolo, collate_fn_moe, model_info, load_checkpoint
from model_training.config import Config
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
                               original_dir=None,
                               save_dir=None,
                               progress_path=None):

    all_config = Config()

    device = all_config.device
    image_transform = all_config.image_transform
    class_color_map = all_config.class_color_map
    transform = all_config.transform
    mask_transform = all_config.mask_transform
    train_image_paths = all_config.train_image_paths
    train_ann_paths = all_config.train_ann_paths
    val_image_paths = all_config.val_image_paths
    val_ann_paths = all_config.val_ann_paths
    test_image_paths = all_config.test_image_paths
    inference_image_paths = all_config.inference_image_paths
    # inference_image_paths=input_inference_path
    test_ann_paths = all_config.test_ann_paths
    img_size = all_config.img_size
    model_dir = all_config.model_dir
    lr = all_config.lr
    weight_decay = all_config.weight_decay
    batch_size = all_config.batch_size
    num_classes = all_config.num_classes
    class_names = all_config.class_names
    image_transform_ema = all_config.image_transform_ema
    mask_transform_ema = all_config.mask_transform_ema
    image_transform_moe = all_config.image_transform_moe
    # mask_transform_moe = all_config.mask_transform_moe
    index_to_classes_dict = all_config.index_to_classes_dict
    resize_img_size = all_config.resize_img_size
    # save_dir = all_config.save_dir

    filename_list = []

    if log_enable:
        log_dir = f"runs/{ml}"
        if os.path.exists(log_dir) and not os.path.isdir(log_dir):
            os.remove(log_dir)
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
    else:
        writer = None

    if ml == "dinov2":
        # Dataset and Dataloader
        train_dataset = SegmentationDataset(img_size, train_image_paths, train_ann_paths, image_transform=image_transform, mask_transform=mask_transform)
        val_dataset = SegmentationDataset(img_size, val_image_paths, val_ann_paths, image_transform=image_transform, mask_transform=mask_transform)
        test_dataset = SegmentationDataset(img_size, test_image_paths, test_ann_paths, image_transform=image_transform, mask_transform=mask_transform)
        inference_dataset = SegmentationDataset(img_size, inference_image_paths, None, image_transform=image_transform, mask_transform=None)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
        inference_loader = DataLoader(inference_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

        steps_per_epoch = len(train_dataloader)

        # Model and Loss
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

        if optimizer_type == "adam":
            optimizer = torch.optim.Adam(trainable_parameters, lr=lr)
        elif optimizer_type == "adamW":
            optimizer = torch.optim.AdamW(trainable_parameters, lr=lr, weight_decay=weight_decay)
        else: # Default mode: adam
            optimizer = torch.optim.Adam(trainable_parameters, lr=lr)

        if scheduler_mode == "cosineanneal":
            scheduler = CosineAnnealingLR(optimizer, T_max=steps_per_epoch * epochs, eta_min=1e-6)
        elif scheduler_mode == "stepLR":
            scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
        elif scheduler_mode == "MultistepLR":
            scheduler = MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)
        elif scheduler_mode == "ExponentialLR":
            scheduler = ExponentialLR(optimizer, gamma=0.95)
        elif scheduler_mode == "ReduceLROnPlateau":
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
        elif scheduler_mode == "onecycleLR":
            scheduler = OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(train_dataloader), epochs=10)
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=steps_per_epoch * epochs, eta_min=1e-6)

        if mode == "train":
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
                }, f"{model_dir}/dinov2_token_segmentation_{epoch+1}.pth")

        else:
            checkpoint = torch.load(f"{model_dir}/dinov2_token_segmentation.pth")
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if mode == "continue_train":
                start_epoch = checkpoint['epoch']
                epochs = start_epoch + epochs

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

            elif mode == "test":
                test_results = test_seg(model, test_loader, device, save_dir=save_dir)

            elif mode == "inference":
                inference_results = inference_seg(model, inference_loader, device, original_dir=original_dir, save_dir=save_dir, progress_path=progress_path)

    elif ml == "yolov9":

        train_dataset = YoloDetectionDataset(train_image_paths, train_ann_paths, transform)
        val_dataset = YoloDetectionDataset(val_image_paths, val_ann_paths, transform)
        test_dataset = YoloDetectionDataset(test_image_paths, test_ann_paths, transform)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_yolo)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_yolo)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn_yolo)

        steps_per_epoch = len(train_dataloader)
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

        if optimizer_type == "adam":
            optimizer = torch.optim.Adam(trainable_parameters, lr=lr)
        elif optimizer_type == "adamW":
            optimizer = torch.optim.AdamW(trainable_parameters, lr=lr, weight_decay=weight_decay)
        else: # Default mode: adam
            optimizer = torch.optim.Adam(trainable_parameters, lr=lr)

        if scheduler_mode == "cosineanneal":
            scheduler = CosineAnnealingLR(optimizer, T_max=steps_per_epoch * epochs, eta_min=1e-6)
        elif scheduler_mode == "stepLR":
            scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
        elif scheduler_mode == "MultistepLR":
            scheduler = MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)
        elif scheduler_mode == "ExponentialLR":
            scheduler = ExponentialLR(optimizer, gamma=0.95)
        elif scheduler_mode == "ReduceLROnPlateau":
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
        elif scheduler_mode == "onecycleLR":
            scheduler = OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(train_dataloader), epochs=10)
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=steps_per_epoch * epochs, eta_min=1e-6)

        if mode == "train":
            # Training loop
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

        elif mode == "continue_train":

            checkpoint = torch.load(f"{model_dir}/yolov9_m4_detection_{epoch}.pth")
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
            test_yolo(model, test_dataloader, device, class_names, save_dir=save_dir)
        
        elif mode == "inference":
            inference_dataset = YoloDetectionDataset(test_image_paths, test_ann_paths, transform)
            inference_dataloader = DataLoader(inference_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn_yolo)
            filename_list = inference_yolo(model, inference_dataloader, device, class_names, save_dir=save_dir)

    elif ml == "mask2former":

        train_dataset = SegmentationDataset_ema(img_size, train_image_paths, train_ann_paths, image_transform=image_transform_ema, mask_transform=mask_transform_ema, box_transform=transform)
        val_dataset = SegmentationDataset_ema(img_size, val_image_paths, val_ann_paths, image_transform=image_transform_ema, mask_transform=mask_transform_ema, box_transform=transform)
        test_dataset = SegmentationDataset_ema(img_size, test_image_paths, test_ann_paths, image_transform=image_transform_ema, mask_transform=mask_transform_ema, box_transform=transform)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

        steps_per_epoch = len(train_dataloader)

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
        
        if optimizer_type == "adam":
            optimizer = torch.optim.Adam(trainable_parameters, lr=lr)
        elif optimizer_type == "adamW":
            optimizer = torch.optim.AdamW(trainable_parameters, lr=lr, weight_decay=weight_decay)
        else: # Default mode: adam
            optimizer = torch.optim.Adam(trainable_parameters, lr=lr)

        if scheduler_mode == "cosineanneal":
            scheduler = CosineAnnealingLR(optimizer, T_max=steps_per_epoch * epochs, eta_min=1e-6)
        elif scheduler_mode == "stepLR":
            scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
        elif scheduler_mode == "MultistepLR":
            scheduler = MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)
        elif scheduler_mode == "ExponentialLR":
            scheduler = ExponentialLR(optimizer, gamma=0.95)
        elif scheduler_mode == "ReduceLROnPlateau":
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
        elif scheduler_mode == "onecycleLR":
            scheduler = OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(train_dataloader), epochs=10)
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=steps_per_epoch * epochs, eta_min=1e-6)
        
        if mode == "train":
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
                    }, f"{model_dir}/mask2former_segmentation_{epoch+1}.pth")
                
        elif mode == "continue_train":

            checkpoint = torch.load(f"{model_dir}/yolov9_m4_detection.pth")
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
                    }, f"{model_dir}/mask2former_segmentation_{epoch+1}.pth")

        elif mode == "test":
            results = inference_maskrcnn_ema(model, test_dataloader, device, class_names=class_names, class_color_map=class_color_map, visualize=False, vis_save_dir=save_dir, conf_thresh=0.5)

    elif ml == "unetr_moe":

        train_dataset = MultiModalSegDataset(img_size, train_image_paths, train_ann_paths, image_transform=image_transform_moe, resize_img_size=resize_img_size, label_map=index_to_classes_dict)
        val_dataset = MultiModalSegDataset(img_size, val_image_paths, val_ann_paths, image_transform=image_transform_moe, resize_img_size=resize_img_size, label_map=index_to_classes_dict)
        test_dataset = MultiModalSegDataset(img_size, test_image_paths, test_ann_paths, image_transform=image_transform_moe, resize_img_size=resize_img_size, label_map=index_to_classes_dict)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_moe)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_moe)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn_moe)

        steps_per_epoch = len(train_dataloader)

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

        if optimizer_type == "adam":
            optimizer = torch.optim.Adam(trainable_parameters, lr=lr)
        elif optimizer_type == "adamW":
            optimizer = torch.optim.AdamW(trainable_parameters, lr=lr, weight_decay=weight_decay)
        else: # Default mode: adam
            optimizer = torch.optim.Adam(trainable_parameters, lr=lr)
        
        if scheduler_mode == "cosineanneal":
            scheduler = CosineAnnealingLR(optimizer, T_max=steps_per_epoch * epochs, eta_min=1e-6)
        elif scheduler_mode == "stepLR":
            scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
        elif scheduler_mode == "MultistepLR":
            scheduler = MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)
        elif scheduler_mode == "ExponentialLR":
            scheduler = ExponentialLR(optimizer, gamma=0.95)
        elif scheduler_mode == "ReduceLROnPlateau":
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
        elif scheduler_mode == "onecycleLR":
            scheduler = OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(train_dataloader), epochs=10)
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=steps_per_epoch * epochs, eta_min=1e-6)

        criterion = CLIP_MultiTaskLoss()

        if mode == "train":
            # Training loop
            # best_dice = 0.0

            #with mlflow.start_run() as run:
            #    run_id = run.info.run_id
            #    print("Run ID:", run_id)

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
                torch.save(model.state_dict(), f"{model_dir}/unetr_moe_best_model_epoch{epoch+1}.pth")

            # mlflow.register_model(model_uri=f"runs:/{run_id}/unetr_moe_model", name="UNETR_MOE_Model")

        elif mode == "continue_train":

            checkpoint = load_checkpoint(model, optimizer, f"{model_dir}/unetr_moe_best_model_epoch{start_epoch}.pth", device)
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
                torch.save(model.state_dict(), f"{model_dir}/unetr_moe_best_model_epoch{epoch+1}.pth")

                #mlflow.register_model(model_uri=f"runs:/{run_id}/unetr_moe_model", name="UNETR_MOE_Model")

        elif mode == "test":
            
            model.load_state_dict(torch.load(f"{model_dir}/unetr_moe_best_model_epoch{start_epoch}.pth"))
            # model = mlflow.pytorch.load_model(f"runs:/{run_id}/unetr_moe_model")
            # checkpoint = load_checkpoint(model, optimizer, f"{model_dir}/unetr_moe_best_model_epoch{start_epoch}.pth", device)
            evaluate_segmentation_model_moe(model, test_dataloader, criterion, device, num_classes=num_classes+1, class_names=class_names, visualize_cm=False)
        
        elif mode == "inference":
            
            if inference_image_paths:
                inference_dataset = MultiModalSegDataset(img_size, inference_image_paths, ann_paths=None, image_transform=image_transform_moe, resize_img_size=resize_img_size, label_map=index_to_classes_dict)
                inference_dataloader = DataLoader(inference_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn_moe)
            else:
                exit(-1)

            model.load_state_dict(torch.load(f"{model_dir}/unetr_moe_best_model_epoch{start_epoch}.pth"))
            # model = mlflow.pytorch.load_model(f"runs:/{run_id}/unetr_moe_model")
            # checkpoint = load_checkpoint(model, optimizer, f"{model_dir}/unetr_moe_best_model_epoch{start_epoch}.pth", device)
            filename_list = inference_segmentation_mode_moe(model, inference_dataloader, device, class_color_map, class_names=class_names, save_dir=save_dir)
    
    return filename_list

