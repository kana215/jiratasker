# 🤖 AI‑секретарь (Vosk → Задачи → Jira)

## Запуск локально
```bash
pip install -r requirements.txt
# Скачай Vosk RU модель и распакуй в models/vosk/<папка_модели>
streamlit run main.py
```

## Хостинг на Render
- Build: `pip install -r requirements.txt`
- Start: `streamlit run main.py --server.port=$PORT --server.address=0.0.0.0`

## Модели
Скачай и распакуй:
- RU: https://alphacephei.com/vosk/models/vosk-model-small-ru-0.22.zip
- EN: https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip

Положи сюда:
```
models/
  └── vosk/
      └── vosk-model-small-ru-0.22/
          ├── am/...
          ├── conf/...
          └── ...
```

(Опционально) Для ИИ‑извлечения задач офлайн:
- FLAN‑T5 small: https://huggingface.co/google/flan-t5-small
Положи в `models/flan/flan-t5-small/`
