"""
Fine-tuning IndoBERT untuk cascade GATE — Google Colab (GPU T4 cukup).
Data dibaca dari Drive: MyDrive/GATESystem/teks_dataset/indobert/{train,val,test}.jsonl
Hasil disimpan permanen ke Drive: MyDrive/GATESystem/indobert_gate/

Cara pakai (Runtime > Change runtime type > T4 GPU), dua sel:
  Sel 1:  !pip -q install "transformers[torch]" scikit-learn
  Sel 2:  copas SELURUH isi file ini, lalu Run.
          (izin akses Drive akan diminta otomatis saat sel berjalan)

Output di MyDrive/GATESystem/indobert_gate/:
  - model + tokenizer terbaik menurut F1 validasi
  - test_predictions.csv (prob per baris test, utk analisis lanjut)
  - test_report.txt
Label: 1 = promosi judol, 0 = bukan promosi.
"""
import json
import os
import sys
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer)
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, classification_report)

IS_COLAB = 'google.colab' in sys.modules
DRIVE_BASE = '/content/drive/MyDrive/GATESystem'

if IS_COLAB and not os.path.exists('/content/drive/MyDrive'):
    from google.colab import drive
    drive.mount('/content/drive')

_default_base = DRIVE_BASE if IS_COLAB else '.'
MODEL_NAME = 'indobenchmark/indobert-base-p1'
DATA_DIR = os.environ.get('DATA_DIR', os.path.join(_default_base, 'teks_dataset', 'indobert'))
OUT_DIR = os.environ.get('OUT_DIR', os.path.join(_default_base, 'indobert_gate'))
MAX_LENGTH = 512
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

# Di luar Colab (lokal) paksa CPU: GPU MPS pada Mac Intel (Radeon ~6,8 GB) OOM
# pada seq 512 dan tidak stabil untuk BERT. use_cpu=True di TrainingArguments
# (di bawah) yang benar-benar menonaktifkannya; env var saja tidak cukup.
# Manfaatkan seluruh thread CPU untuk mempercepat.
FORCE_CPU = not IS_COLAB
if FORCE_CPU:
    torch.set_num_threads(os.cpu_count() or 8)


def load_jsonl(name):
    path = os.path.join(DATA_DIR, f'{name}.jsonl')
    if not os.path.exists(path):
        sys.exit(f'[ERROR] {path} tidak ditemukan. Pastikan train/val/test.jsonl '
                 f'sudah di-upload ke {DATA_DIR}/')
    with open(path, encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]


class TextDataset(Dataset):
    def __init__(self, rows, tokenizer):
        self.rows = rows
        self.tok = tokenizer

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        enc = self.tok(r['text'], truncation=True, max_length=MAX_LENGTH,
                       padding='max_length', return_tensors='pt')
        return {'input_ids': enc['input_ids'].squeeze(0),
                'attention_mask': enc['attention_mask'].squeeze(0),
                'labels': torch.tensor(r['label'])}


class WeightedTrainer(Trainer):
    """CrossEntropy berbobot kelas: dataset timpang ~2,7:1 (judol:non)."""

    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop('labels')
        outputs = model(**inputs)
        loss = nn.CrossEntropyLoss(weight=self.class_weights.to(outputs.logits.device))(
            outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average='binary',
                                                  pos_label=1, zero_division=0)
    return {'accuracy': accuracy_score(labels, preds),
            'precision': p, 'recall': r, 'f1': f1}


def main():
    train_rows = load_jsonl('train')
    val_rows = load_jsonl('val')
    test_rows = load_jsonl('test')
    print(f'train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}')

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    n = len(train_rows)
    n_pos = sum(r['label'] for r in train_rows)
    weights = torch.tensor([n / (2 * (n - n_pos)), n / (2 * n_pos)], dtype=torch.float)
    print('bobot kelas [non, judol]:', weights.tolist())

    # Batch 8 (aman utk 16 GB RAM di CPU) + akumulasi gradien 2 => efektif 16,
    # sesuai learning rate. Checkpoint per epoch disimpan agar run panjang bisa
    # dilanjutkan bila terputus (resume otomatis di bawah).
    ckpt_dir = '/content/checkpoints' if IS_COLAB else './checkpoints'
    on_gpu = torch.cuda.is_available()
    args = TrainingArguments(
        output_dir=ckpt_dir,
        num_train_epochs=3,
        per_device_train_batch_size=16 if on_gpu else 8,
        gradient_accumulation_steps=1 if on_gpu else 2,
        per_device_eval_batch_size=32 if on_gpu else 8,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        logging_steps=25,
        fp16=on_gpu,
        use_cpu=FORCE_CPU,
        dataloader_pin_memory=on_gpu,
        seed=SEED,
        report_to='none',
    )

    trainer = WeightedTrainer(
        class_weights=weights,
        model=model,
        args=args,
        train_dataset=TextDataset(train_rows, tokenizer),
        eval_dataset=TextDataset(val_rows, tokenizer),
        compute_metrics=compute_metrics,
    )
    # lanjutkan dari checkpoint terakhir bila ada (mis. setelah training terputus)
    resume = os.path.isdir(ckpt_dir) and any(
        d.startswith('checkpoint-') for d in os.listdir(ckpt_dir))
    if resume:
        print(f'[INFO] Melanjutkan dari checkpoint di {ckpt_dir}')
    trainer.train(resume_from_checkpoint=resume)

    os.makedirs(OUT_DIR, exist_ok=True)
    trainer.save_model(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)

    # evaluasi test set + simpan probabilitas per baris
    pred = trainer.predict(TextDataset(test_rows, tokenizer))
    probs = torch.softmax(torch.tensor(pred.predictions), dim=-1)[:, 1].numpy()
    labels = np.array([r['label'] for r in test_rows])
    preds = (probs >= 0.5).astype(int)

    report = classification_report(labels, preds,
                                   target_names=['bukan-promosi', 'promosi-judol'],
                                   digits=4)
    cm = confusion_matrix(labels, preds)
    print(report)
    print('confusion matrix [baris=aktual, kolom=prediksi]:\n', cm)
    with open(os.path.join(OUT_DIR, 'test_report.txt'), 'w') as f:
        f.write(report + '\n' + str(cm) + '\n')

    with open(os.path.join(OUT_DIR, 'test_predictions.csv'), 'w') as f:
        f.write('url,label,prob_judol\n')
        for r, p in zip(test_rows, probs):
            f.write(f"{r['url']},{r['label']},{p:.4f}\n")
    print(f'Selesai. Model & laporan di {OUT_DIR}/')


if __name__ == '__main__':
    main()
