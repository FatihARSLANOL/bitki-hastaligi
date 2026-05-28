# bitki-hastaligi
Bu proje, ResNet50 mimarisini [PlantVillage](https://www.kaggle.com/datasets/emmarex/plantdisease) veri seti üzerinde ince ayar yaparak (fine-tuning) bitki yapraklarındaki hastalıkları sınıflandırmayı amaçlamaktadır. PyTorch framework'ü kullanılarak geliştirilmiştir.

---

## Sonuçlar

Modelin test seti üzerindeki performansı aşağıdaki gibidir:

| Metrik | Değer |
|---|---|
| Test Doğruluğu | **%98.79** |
| Ağırlıklı F1 | **0.9879** |
| Makro F1 | **0.9857** |
| En İyi Val. Doğruluğu | **0.9908** |

### Sınıf Bazında F1 Skorları

| Sınıf | F1 |
|---|---|
| Pepper\_\_bell\_\_\_Bacterial\_spot | 0.9950 |
| Pepper\_\_bell\_\_\_healthy | 0.9966 |
| Potato\_\_\_Early\_blight | 1.0000 |
| Potato\_\_\_Late\_blight | 0.9900 |
| Potato\_\_\_healthy | 0.9677 |
| Tomato\_Bacterial\_spot | 0.9858 |
| Tomato\_Early\_blight | 0.9552 |
| Tomato\_Late\_blight | 0.9844 |
| Tomato\_Leaf\_Mold | 0.9733 |
| Tomato\_Septoria\_leaf\_spot | 0.9830 |
| Tomato\_Spider\_mites\_Two\_spotted\_spider\_mite | 0.9851 |
| Tomato\_\_Target\_Spot | 0.9892 |
| Tomato\_\_Tomato\_YellowLeaf\_\_Curl\_Virus | 0.9938 |
| Tomato\_\_Tomato\_mosaic\_virus | 0.9870 |
| Tomato\_healthy | 1.0000 |

### Eğitim Grafikleri

![Eğitim Geçmişi](bitki%20hastaligi/results/history.png)

### Karmaşıklık Matrisi

![Karmaşıklık Matrisi](bitki%20hastaligi/results/confusion_matrix.png)

### Tahminler

![Karmaşıklık Matrisi](bitki%20hastaligi/results/predictions.png)

---

## Literatür içerisinde

PlantVillage veri seti, kontrollü laboratuvar koşullarında (düz arka plan, sabit aydınlatma) çekilmiş görüntülerden oluştuğu için akademik çalışmalarda tutarlı biçimde yüksek doğruluk oranları elde edilmektedir.

> **Not:** PlantVillage veri setinin bilinen bir kısıtlaması, gerçek saha fotoğraflarında genelleme performansının düşebilmesidir. Laboratuvar koşullarında eğitilen modeller, sahada çekilen görüntülerde genellikle %60–70 civarında doğruluk sergilemektedir.

---

## Veri Seti

- **Kaynak:** [PlantVillage — Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease)
- **Toplam görüntü:** 20.638
- **Sınıf sayısı:** 15
- **Kapsanan bitkiler:** Biber, Patates, Domates
- **Bölünme:** %80 eğitim / %10 doğrulama / %10 test (tabakalı örnekleme ile)

---

## Model Mimarisi

Önceden ImageNet üzerinde eğitilmiş **ResNet50** modeli, özelleştirilmiş bir sınıflandırma başlığı ile ince ayara tabi tutulmuştur.

```
ResNet50 (ImageNet ağırlıkları)
    └── Backbone (dondurulmuş → epoch 4'te layer4 açıldı)
    └── FC Katmanı:
            Linear(2048 → 512)
            ReLU
            Dropout(0.4)
            Linear(512 → 15)
```

**İki aşamalı eğitim stratejisi:**
- Epoch 1–3 → Yalnızca yeni başlık eğitildi, backbone donduruldu
- Epoch 4–10 → `layer4` + başlık birlikte ince ayara tabi tutuldu

---

## Eğitim Süreci

| Parametre | Değer |
|---|---|
| Optimizer | SGD + Momentum |
| Momentum | 0.9 |
| Scheduler | StepLR |
| Öğrenme Oranı | 0.01 → epoch 4'te 0.001 |
| Ağırlık Azalması | 5e-4 |
| Batch Boyutu | 32 |
| Epoch Sayısı | 10 |
| Görüntü Boyutu | 224 × 224 |

### Veri Artırma (yalnızca eğitim)
- 256×256'ya yeniden boyutlandırma ardından rastgele kırpma
- Rastgele yatay çevirme
- Gaussian bulanıklaştırma
- Rastgele gri tonlamaya dönüştürme (%5 olasılıkla)
- ImageNet normalizasyonu

---

Tahmin çıktısı her görüntü için en olası 3 sınıfı güven yüzdesiyle birlikte gösterir ve `results/predictions.png` olarak kaydeder.

> **İpucu:** Model, PlantVillage tarzı görüntülerde (düz arka plan, izole yaprak) en iyi performansı gösterir. Gerçek saha fotoğraflarında doğruluk düşebilir.

---

## Gereksinimler

```
torch
torchvision
scikit-learn
matplotlib
seaborn
tqdm
```

---

## Kaynaklar

- [Deep Residual Learning for Image Recognition (ResNet)](https://arxiv.org/abs/1512.03385)
- [PlantVillage Veri Seti Makalesi](https://arxiv.org/abs/1511.08060)
- [Derin Öğrenme ile Bitki Hastalığı Tespiti](https://www.frontiersin.org/articles/10.3389/fpls.2016.01419/full)
