import os
import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import io

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class ModelTrainer:
    async def train_model(self, data: pd.DataFrame, bot, chat_id) -> float:

        try:
            if data.empty:
                logger.error("Загруженный DataFrame пуст. Проверьте содержимое файла.")
                await bot.send_message(chat_id, "Ошибка: загруженный файл пуст.")
                return None

            logger.info("Первые 5 строк данных:")
            logger.info(data.head())
            logger.info("Количество NaN в каждом столбце до обработки:")
            logger.info(data.isna().sum())
            logger.info("Типы данных в каждом столбце:")
            logger.info(data.dtypes)

            numerical_columns = data.select_dtypes(include=['number']).columns
            categorical_columns = data.select_dtypes(exclude=['number']).columns

            if len(numerical_columns) == 0:
                logger.error("Нет числовых колонок в данных. Обучение модели невозможно.")
                await bot.send_message(chat_id, "Ошибка: нет числовых колонок в данных.")
                return None

            if len(categorical_columns) == 0:
                logger.error("Нет категориальных колонок в данных. Обучение модели невозможно.")
                await bot.send_message(chat_id, "Ошибка: нет категориальных колонок в данных.")
                return None

            imputer = SimpleImputer(strategy='mean')
            data[numerical_columns] = imputer.fit_transform(data[numerical_columns])

            data.dropna(inplace=True)

            if data.empty:
                logger.error("Все данные были удалены из-за NaN. Проверьте входные данные.")
                await bot.send_message(chat_id, "Ошибка: все данные были удалены из-за NaN.")
                return None

            X = data.iloc[:, :-1]
            y = data.iloc[:, -1]

            numerical_features = X.select_dtypes(include=['number']).columns
            categorical_features = X.select_dtypes(exclude=['number']).columns

            X_numerical = imputer.fit_transform(X[numerical_features])

            if categorical_features.size > 0:
                encoder = OneHotEncoder(sparse_output=False,
                                        handle_unknown='ignore')
                X_categorical = encoder.fit_transform(X[categorical_features])
            else:
                X_categorical = np.array([]).reshape(0, 0)
                encoder = None

            if X_numerical.size == 0 and X_categorical.size == 0:
                logger.error("Нет данных для обучения. Проверьте входные данные.")
                await bot.send_message(chat_id, "Ошибка: нет данных для обучения.")
                return None

            if X_categorical.size > 0:
                X_processed = np.hstack((X_numerical, X_categorical))
            else:
                X_processed = X_numerical

            X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
            logger.info(f"Размеры обучающей выборки: {len(X_train)}, тестовой выборки: {len(X_test)}")

            models = {
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
            }

            cross_val_scores_list = []
            confusion_matrices = []
            feature_importances = []

            for name, model in models.items():
                cross_val_scores = cross_val_score(model, X_train, y_train, cv=5)
                cross_val_mean_accuracy = np.mean(cross_val_scores)
                logger.info(f"{name} - Средняя точность кросс-валидации: {cross_val_mean_accuracy:.2f}")

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                test_accuracy = accuracy_score(y_test, y_pred)
                logger.info(f"{name} - Точность на тестовой выборке: {test_accuracy:.2f}")

                cross_val_scores_list.append(cross_val_scores)
                confusion_matrices.append(confusion_matrix(y_test, y_pred))
                if hasattr(model, 'feature_importances_'):
                    feature_importances.append(model.feature_importances_)

            for scores in cross_val_scores_list:
                await self.send_cross_val_plot(scores, bot, chat_id)

            for index, cm in enumerate(confusion_matrices):
                y_test_pred = models[list(models.keys())[index]].predict(X_test)
                await self.send_confusion_matrix_plot(y_test, y_test_pred, bot, chat_id)

            if encoder is not None:
                for importance in feature_importances:
                    await self.send_feature_importance_plot(importance, numerical_features, categorical_features, encoder, bot, chat_id)

            return test_accuracy

        except Exception as e:
            logger.error(f"Ошибка при обучении модели: {str(e)}")
            await bot.send_message(chat_id, f"Ошибка при обучении модели: {str(e)}")
            return None

    async def send_cross_val_plot(self, scores, bot, chat_id):
        plt.figure()
        plt.plot(scores)
        plt.title("Scores для кросс-валидации")
        plt.xlabel("Fold")
        plt.ylabel("Accuracy")
        plt.grid()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        await bot.send_photo(chat_id, photo=buf)
        plt.close()
        buf.close()

    async def send_confusion_matrix_plot(self, y_test, y_pred, bot, chat_id):
        cm = confusion_matrix(y_test, y_pred)
        plt.figure()
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Матрица ошибок")
        plt.colorbar()
        tick_marks = np.arange(len(set(y_test)))
        plt.xticks(tick_marks, tick_marks)
        plt.yticks(tick_marks)
        plt.xlabel('Предсказанные метки')
        plt.ylabel('Истинные метки')
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        await bot.send_photo(chat_id, photo=buf)
        plt.close()
        buf.close()

    async def send_feature_importance_plot(self, feature_importances, numerical_features, categorical_features, encoder, bot, chat_id):
        if feature_importances is not None:
            feature_names = np.concatenate([numerical_features, encoder.get_feature_names_out(categorical_features)])
            indices = np.argsort(feature_importances)[::-1][:10]  

            plt.figure(figsize=(10, 6))
            plt.title("Топ-10 признаков по важности")
            plt.bar(range(10), feature_importances[indices], align='center')
            plt.xticks(range(10), feature_names[indices], rotation=90)
            plt.xlim([-1, 10])
            plt.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            await bot.send_photo(chat_id, photo=buf)
            plt.close()
            buf.close()
