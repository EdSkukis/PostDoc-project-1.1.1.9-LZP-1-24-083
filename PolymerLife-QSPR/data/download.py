import os
import shutil
import pandas as pd
import kagglehub
from utils.logger import logger


class DatasetDownloader:
    def __init__(self, dataset_handle="ko55584/extended-polymer-dataset", target_dir="data/raw"):
        self.dataset_handle = dataset_handle
        self.target_dir = target_dir
        self.target_file = os.path.join(self.target_dir, "extended_polymer_dataset.csv")

    def download_and_verify(self) -> pd.DataFrame:
        """
        Downloads, checks for updates, and validates the dataset.
        Returns a pandas DataFrame.
        """
        os.makedirs(self.target_dir, exist_ok=True)

        try:
            logger.info(f"Checking for dataset updates on Kaggle: {self.dataset_handle}...")

            # kagglehub автоматически проверяет наличие новой версии на сервере.
            # Если новой версии нет, он просто возвращает путь к локальному кэшу.
            cache_path = kagglehub.dataset_download(self.dataset_handle)
            logger.info(f"Kaggle cache successfully processed: {cache_path}")

            # Ищем CSV файл в папке кэша (имя файла может отличаться от названия датасета)
            csv_file = None
            for file in os.listdir(cache_path):
                if file.endswith(".csv"):
                    csv_file = os.path.join(cache_path, file)
                    break

            if not csv_file:
                raise FileNotFoundError("No file with extension was found in the downloaded dataset. .csv")

            # Сравниваем кэш с нашим рабочим файлом в data/raw/
            needs_update = True
            if os.path.exists(self.target_file):
                # Сверяем размеры файлов. Если равны — считаем, что обновления нет
                cache_size = os.path.getsize(csv_file)
                local_size = os.path.getsize(self.target_file)

                if cache_size == local_size:
                    logger.info("The local file in data/raw/ is up to date (no changes).")
                    needs_update = False
                else:
                    logger.info("A new or modified version of the dataset was found. Updating...")

            # Копируем файл из кэша в рабочую папку проекта, если нужно
            if needs_update:
                shutil.copy2(csv_file, self.target_file)
                logger.info(f"File successfully copied/updated to: {self.target_file}")

        except Exception as e:
            logger.warning(f"Error accessing Kaggle or network: {str(e)}")
            if os.path.exists(self.target_file):
                logger.info("Error accessing Kaggle or Switching to OFFLINE MODE: Using a previously downloaded local version of the network.")
            else:
                logger.critical("CRITICAL ERROR: There is no local copy and the data could not be downloaded.")
                raise RuntimeError("Unable to obtain data to run the framework.") from e

        logger.info("Reading and checking the dataset structure...")
        try:
            df = pd.read_csv(self.target_file)
            logger.info(f"The dataset has been successfully loaded into memory. Full size: {df.shape}")

            # Проверка наличия критически важных для нас колонок
            if 'SMILES' not in df.columns or 'Tc' not in df.columns:
                logger.error(
                    "The required columns 'SMILES' or 'Tc' are missing. The dataset structure may have changed!")
            else:
                valid_rows = df[['SMILES', 'Tc']].dropna().shape[0]
                logger.info(f"Rows with training data (SMILES + Tc): {valid_rows}")

            return df

        except pd.errors.EmptyDataError:
            logger.critical("The data file is empty or corrupted!")
            raise
        except Exception as e:
            logger.critical(f"Unexpected error reading CSV: {str(e)}")
            raise


if __name__ == "__main__":
    downloader = DatasetDownloader()
    df_raw = downloader.download_and_verify()