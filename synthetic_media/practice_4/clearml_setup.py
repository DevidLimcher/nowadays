from clearml import Task

def initialize_clearml():
    # Отключаем автоматическое отслеживание моделей и артефактов
    task = Task.init(
        project_name="Speaker Verification",
        task_name="Training",
        auto_connect_frameworks={"pytorch": False},
        auto_connect_arg_parser=False,
        auto_connect_streams=False
    )

    # Отключаем автоматическое отслеживание артефактов
    task.set_base_task(None)

    print("ClearML initialized with custom settings.")
    return task
