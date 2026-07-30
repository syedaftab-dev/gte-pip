## Technical Explanation — Hinglish mein 🧠

---

### 🔴 Mode Collapse / Class Imbalance

Dataset mein **84% negative (non-binding)** aur **16% positive (binding)** residues hain.

`CrossEntropyLoss` by default har sample ko equal weight deta hai. Toh model sikhta hai:
> "Agar main **har cheez ko 0 (non-binding) predict karun**, toh mera loss automatically 84% accuracy de dega."

Is problem ko **mode collapse** kehte hain — model ek hi class predict karna seekh leta hai.

---

### 🟡 Weighted CrossEntropyLoss

Formula:
```
Loss = -[w₀ · y=0 · log(p₀) + w₁ · y=1 · log(p₁)]
```

Jab `w₁ = neg/pos = 5.52` set karo — matlab class-1 (binding) miss karna class-0 miss karne se **5.52x zyada costly** ho jaata hai. Lekin **problem yeh thi**: weight bahut aggressive tha, model dusri extreme pe chala gaya — har cheez ko positive predict karne laga.

---

### 🟢 Focal Loss

Weighted loss se better. Formula:

```
FL(p_t) = -(1 - p_t)^γ · log(p_t)
```

Yahan `p_t = probability of correct class`.

- Agar model **confident hai** (easy sample) → `p_t` high → `(1-p_t)^γ ≈ 0` → **loss almost zero**
- Agar model **confident nahi** (hard sample) → `p_t` low → `(1-p_t)^γ ≈ 1` → **loss full**

`γ=2` matlab: easy negatives (jo model already sahi predict kar raha) training mein **contribute nahi karte**. Model ka **focus automatically hard positives pe shift** ho jaata hai.

---

### 🔵 sqrt(neg/pos) Class Weight

Raw ratio `neg/pos = 5.52` — bahut extreme.

Iski jagah `sqrt(5.52) = 2.35` use karte hain. Yeh ek **softer interpolation** hai between "equal weights" (1.0) aur "full inverse frequency" (5.52).

```
pos_weight = (neg/pos)^0.5  →  between 1.0 and 5.52
```

---

### 🟣 Threshold Search Bug (0.0)

Model softmax output deta hai `p ∈ [0, 1]`.

Binary prediction ke liye threshold `t` use karte hain:
```
pred = 1  if p >= t  else 0
```

Humara purana code `range(0, 100)` se start karta tha, matlab `t = 0.0` bhi try karta tha.

`t = 0.0` matlab `p >= 0.0` — **har softmax output ≥ 0** hota hai, toh **sabka prediction = 1 (positive)**.

Class imbalance ke saath `recall = 1.0`, `precision ≈ 0.156`, `F1 ≈ 0.27`.

Yeh F1 = 0.27 **optimal** lag raha tha, isliye algorithm `threshold = 0.0` return karta tha. Fix: `range(1, 100)` → `t` starts from `0.01`.

---

### 🟠 AUC vs AUPRC — Kya farq hai?

**AUC (ROC):**
```
TPR = TP / (TP + FN)   [recall]
FPR = FP / (FP + TN)
```
AUC = area under TPR vs FPR curve. **Imbalanced data mein misleading** — FPR bahut chhota hota hai aise datasets mein.

**AUPRC (Precision-Recall):**
```
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
```
AUPRC = area under Precision vs Recall curve. **Imbalanced datasets ke liye sahi metric** — positives pe directly focus karta hai. Yahi reason hai ki hum AUPRC pe model save/early stopping karte hain.

---

### 🔶 CosineAnnealingLR

Learning Rate `η` ko epochs ke saath smooth tarike se decay karta hai:

```
η_t = η_min + 0.5 × (η_max - η_min) × (1 + cos(π × t / T_max))
```

`t` = current epoch, `T_max` = total epochs.

Pahle tha `ReduceLROnPlateau` — yeh tab reduce karta hai jab metric improve nahi hota. Lekin jab AUPRC = 0 (mode collapse), yeh **kabhi trigger nahi hota** aur LR high reh jaata. Cosine schedule **independent** hai metric se — hamesha decay karta hai.

---

### 🟤 Early Stopping

Har epoch pe validation AUPRC track karta hai. Agar `patience=15` epochs tak koi improvement nahi → training rok deta hai.

Best checkpoint save hota hai jab bhi AUPRC improve karta hai — final model wahi use hota hai, na ki last epoch wala.