from model_archive.main_entry import model_trainvaltest_process

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

    question = "Is there any precancerous area? Do I need to go to clinic for further checkup?"

    model_trainvaltest_process(
        optimizer_type="adam",
        lr=1e-4,
        scheduler_mode="cosineanneal",
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
        db_path=None,
        question=question
    )