# Стартовий шаблон

Інструментарій для ДЗ — **тільки те, що шкідливо змушувати винаходити**. Решту пишете самі.

## Що тут є

| Файл | Що робить | Чи треба чіпати |
|---|---|---|
| `data_loader.py` | Streaming MS MARCO, правильна збірка relevant + distractors, reproducible subsets з seed | Ні (можна, якщо хочете інший датасет) |
| `metrics.py` | `recall@k`, `mrr@k`, `evaluate()` — фіксована імплементація | Ні (щоб усі рахували однаково) |
| `requirements.txt` | Мінімум залежностей | Доповнюйте під свій fix |

## Що написати самим

1. **Embedding функцію** — модель, batch size, device (CPU/MPS/CUDA), instruction prefix
2. **Retriever** — інтерфейс і реалізації (naive numpy → FAISS HNSW / Qdrant / hybrid BM25+dense+RRF / reranker)
3. **Scaling loop** — для кожного `size` зібрати subset → проіндексувати → search → виміряти час → зберегти CSV
4. **Cost і RAM tracking** — як саме міряти і де записувати
5. **Візуалізацію** — matplotlib або seaborn, графіки scaling + baseline-vs-fix

## Як запустити

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# плюс додайте свої залежності: sentence-transformers, faiss-cpu, openai тощо

# 1) один раз — кешувати корпус (~5-10 хв streaming)
python data_loader.py

# 2) ваш scaling скрипт — пишете самі
python your_experiment.py
```

## Контракти

**Eval set формат** (з `data_loader.py`):
```python
[{"qid": "12345", "query": "what is X", "relevant_ids": ["doc_42", "doc_7"]}]
```

**Metrics виклик** (з `metrics.py`):
```python
from metrics import evaluate
results = evaluate(eval_set, retrieved_per_query, ks=(1, 5, 10))
# {"recall@1": 0.7, "recall@5": 0.93, "recall@10": 0.93, "mrr@10": 0.80}
```

`retrieved_per_query` — це `list[list[doc_id]]`: для кожного запиту в `eval_set` ваш ranked top-K.

## Чому шаблон такий мінімальний

- **Streaming MS MARCO** залишено, бо там легко провалитись — взяти перші 100K passages → recall = 0, тому що qrels посилаються на docs з усього 8.84M корпусу
- **Метрики** залишено, бо інакше один студент рахує rank from 0, інший from 1, і результати непорівнянні
- **Retriever і visualization** не залишено навмисно — це і є серце ДЗ. Хай ваш дизайн API і ваш стиль графіків будуть вашими
