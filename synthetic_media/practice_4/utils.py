# Получение ID спикера из имени файла
def get_speaker_id_from_file(file_path):
    return file_path.split('/')[-2]  # Возвращает ID спикера (например, p225)
