AI (Artificial Intelligence, Искусственный интеллект)
│
├── Machine Learning (ML, Машинное обучение)
│   │  → учим алгоритмы на данных, чтобы они находили закономерности
│   │  → используется: рекомендации, прогнозы, распознавание объектов
│   │  → язык: Python (основной), R, иногда Julia
│   │
│   ├── Classical ML (Классическое машинное обучение)
│   │   │
│   │   ├── Линейная/логистическая регрессия → прогнозы, классификация
│   │   ├── SVM (Support Vector Machines) → классификация, распознавание текста/изображений
│   │   ├── Decision Trees / Random Forests / XGBoost → табличные данные, финтех, медицина
│   │   ├── K-means, DBSCAN → кластеризация (поиск групп в данных)
│   │   └── Инструменты: scikit-learn, LightGBM, CatBoost
│   │
│   └── Neural Networks (Нейронные сети)
│       │  → метод в ML, вдохновлён работой мозга
│       │  → язык: Python (через PyTorch/TensorFlow), иногда C++ для продакшена
│       │
│       ├── Deep Learning (Глубокое обучение, DL)
│       │   │  → нейросети с большим числом слоёв, основной драйвер ИИ сегодня
│       │   │  → используется: компьютерное зрение, речь, LLM, генеративные модели
│       │   │
│       │   ├── Computer Vision (CV, Компьютерное зрение)
│       │   │   │  → распознавание лиц, машинное зрение, автопилоты
│       │   │   └── CNN (Convolutional Neural Networks) – свёрточные сети
│       │   │       │  → PyTorch, TensorFlow, OpenCV
│       │   │
│       │   ├── Speech / Audio Processing (Обработка речи и аудио)
│       │   │   │  → распознавание речи, синтез голоса
│       │   │   └── RNN / LSTM / Transformer-based модели
│       │   │
│       │   ├── NLP (Natural Language Processing)
│       │   │   │  → работа с текстами: переводчики, чат-боты, поиск
│       │   │   │
│       │   │   ├── Embeddings (Эмбеддинги)
│       │   │   │   → превращение текста/картинок/аудио в векторы чисел
│       │   │   │   → используется в поиске, кластеризации, RAG, рекомендательных системах
│       │   │   │   → библиотеки: sentence-transformers, OpenAI Embeddings
│       │   │   │
│       │   │   ├── RAG (Retrieval-Augmented Generation)
│       │   │   │   → комбинация поиска по данным + LLM для ответа
│       │   │   │   → пример: чат-бот, который отвечает на основе твоей базы документов
│       │   │   │   → инструменты: LangChain, LlamaIndex
│       │   │   │
│       │   │   └── LLM (Large Language Models)
│       │   │       │  → большие языковые модели (GPT, LLaMA, Claude, Mistral)
│       │   │       │  → используются для генерации текста, кода, анализа данных
│       │   │       │  → тренируются на Python (PyTorch), но API доступны на любых языках
│       │   │       │
│       │   │       ├── Chatbots → ChatGPT, Claude
│       │   │       ├── Code Models → Codex, CodeLlama
│       │   │       └── Multimodal Models (текст+картинки+аудио)
│       │   │
│       │   └── Generative Models (Генеративные модели)
│       │       │  → создание нового контента
│       │       │
│       │       ├── GANs (Generative Adversarial Networks)
│       │       │   → генерация изображений, deepfake
│       │       ├── Diffusion Models
│       │       │   → Stable Diffusion, MidJourney, DALL·E
│       │       └── Music/Video generation → AudioLM, Sora
│       │
│       └── Reinforcement Learning (RL, Обучение с подкреплением)
│           │  → обучение через взаимодействие с окружением
│           │  → используется: игры (AlphaGo), роботы, оптимизация
│           │  → инструменты: Gym, RLlib
│
├── Data Science (Наука о данных)
│   │  → область/профессия, объединяющая математику, статистику и ML
│   │  → язык: Python, R
│   │
│   ├── Data Analysis (Анализ данных)
│   │   → отчёты, дашборды, SQL, визуализация
│   │   → инструменты: Pandas, Excel, Tableau, PowerBI
│   │
│   ├── Exploratory Data Analysis (EDA)
│   │   → исследование данных, статистика, визуализация
│   │   → matplotlib, seaborn, plotly
│   │
│   └── Big Data / Data Engineering
│       → работа с огромными объёмами данных
│       → Hadoop, Spark, Kafka
│
└── Инфраструктура и фреймворки вокруг AI
    │
    ├── Hugging Face → хранилище моделей и датасетов (Python)
    ├── LangChain → связка LLM с БД, API, памятью (Python/JS)
    ├── LlamaIndex → похож на LangChain, заточен под RAG (Python)
    ├── ONNX → перенос моделей между фреймворками (C++/Python)
    └── Deployment (выкатка моделей)
        ├── FastAPI / Flask → API для ML моделей (Python)
        ├── Docker / Kubernetes → контейнеризация и масштабирование
        └── Triton Inference Server → оптимизация и инференс на GPU
