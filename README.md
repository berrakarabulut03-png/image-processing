# Image Processing

TEKNOFEST 2026 otonom araçlar için geliştirilen görüntü işleme ve haritalama modülleri: YOLO tabanlı nesne tespiti, klasik OpenCV ile hedef tespiti, derinlik kamerasıyla mesafe ölçümü ve lidar tabanlı yerel haritalama.

## Sonuçlar

| Model / Modül | Yöntem | Sonuç |
|---|---|---|
| Parkur tabelası tanıma (13 sınıf) | YOLO | **mAP50 0.995** |
| Trafik konisi tespiti + mesafe | YOLO + pinhole mesafe kestirimi | **mAP50 0.934** |
| Atış hedefi merkezi (konsantrik halka) | Klasik OpenCV, kontur tabanlı | 3/3 test |
| Şamandıra tespiti + mesafe | RealSense renk + derinlik, OpenCV | — |

## Modüller

- **`sensors/tabela_detector.py`**: Parkur tabelalarını YOLO ile tanır. Küçük/uzak kutuları eler, yanlış okumanın görevi bozmaması için yüksek güven eşiği kullanır.
- **`sensors/cone_detector.py`**: Koni tespiti. Bilinen koni genişliği ve kamera görüş açısıyla pinhole modelinden mesafe kestirir, mantıksız mesafeleri reddeder.
- **`sensors/target_detector.py`**: Atış hedefinin alt-piksel merkezini YOLO kullanmadan bulur: Otsu eşikleme → dairesel konturlar → aynı merkezli, farklı yarıçaplı halka grubu. HoughCircles bilinçli olarak kullanılmadı (aynı merkezli daireleri bastırır).
- **`sensors/camera_node.py`**: Intel RealSense D455F ile şamandıra tespiti. Derinlik 0 geldiğinde 5×5 komşuluk medyanıyla sağlam mesafe ölçümü, morfolojik filtreleme ve başlangıç ısınması içerir.
- **`comm/goruntu_stream.py`**: Kamera görüntülerini UDP ile aktarır.
- **`models/`**: Eğitilmiş `tabela_best.pt` ve `cone_best.pt`.

## Haritalama

- **`mapping/costmap.py`**: Araç merkezli 2B maliyet haritası (costmap). Engelleri sınıflarına göre (koni, kayar engel, bariyer) farklı güvenlik paylarıyla işler, haritayı yer istasyonuna aktarılmak üzere serileştirir.
- **`mapping/obstacle_classifier.py`**: Lidar taramasını kümelere ayırıp genişliğe göre engel sınıfı tahmin eder. Ardışık taramalar arasında aracın kendi hareketini telafi ederek hareketli engellerin hızını kestirir.
- **`mapping/occupancy_map.py`**: Araç merkezli yerel doluluk haritası (occupancy map) görüntüleyici. Engeller birikir ve zamanla sönümlenir.

## Çalıştırma

```
python -m venv venv
venv\Scripts\activate          # Linux: source venv/bin/activate
pip install -r requirements.txt

python camera_test.py       # webcam ile tabela + koni testi
python samandira_test.py    # RealSense ile şamandıra testi
```

## Benim rolüm

<!-- Kendi katkını buraya yaz. -->
