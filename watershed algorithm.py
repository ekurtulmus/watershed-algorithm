print("HAVZA ALGORİTMASI")
"Bir görüntüdeki farklı nesneleri ayırmak için kullanılan klasik bir yöntemdir"

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Görüntüyü içe aktar
coin = cv2.imread("coins.jpg")
plt.figure(), plt.imshow(coin), plt.axis("off")
# coins.jpg dosyasını içe aktararak görüntüyü yükler.

# Düşük geçiş filtresi: bulanıklaştırma
coin_blur = cv2.medianBlur(coin, 13)
plt.figure(), plt.imshow(coin_blur), plt.axis("off")
# Görüntüyü median blur ile bulanıklaştırarak gürültüyü azaltır.

# Gri tonlamaya çevir
coin_gray = cv2.cvtColor(coin_blur, cv2.COLOR_BGR2GRAY)
plt.figure(), plt.imshow(coin_gray, cmap="gray"), plt.axis("off")
# Bulanıklaştırılmış görüntüyü gri tonlamaya çevirir.

# İkili eşikleme (arkadaki farkı daha belirgin hale getirme, arka siyah ön beyaz)
ret, coin_thresh = cv2.threshold(coin_gray, 75, 255, cv2.THRESH_BINARY)
plt.figure(), plt.imshow(coin_thresh, cmap="gray"), plt.axis("off")
# Görüntüyü ikili eşikleme ile siyah-beyaz hale getirir, arka plan siyah, nesneler beyaz olur.

# Kontur bulma
# contours, hierarchy = cv2.findContours(coin_thresh.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
contours, hierarchy = cv2.findContours(coin_thresh.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
# İkili eşiklenmiş görüntüdeki konturları ve hiyerarşiyi bulur.

# Her bir konturu çizin
for i in range(len(contours)):
    if hierarchy[0][i][3] == -1:  # En dış kontur
        cv2.drawContours(coin, contours, i, (0, 255, 0), 10)
plt.figure(), plt.imshow(coin), plt.axis("off")
# Hiyerarşide dış kontur olanları bulup, yeşil renkte kalın çizgilerle konturları çizer.

# Watershed algoritması

# Görüntüyü içe aktar
coin = cv2.imread("coins.jpg")
plt.figure(), plt.imshow(coin), plt.axis("off")
# Görüntüyü tekrar yükler (çizilen konturları kaldırmak için).

# Düşük geçiş filtresi: bulanıklaştırma
coin_blur = cv2.medianBlur(coin, 13)
plt.figure(), plt.imshow(coin_blur), plt.axis("off")
# Görüntüyü tekrar median blur ile bulanıklaştırır.

# Gri tonlamaya çevir
coin_gray = cv2.cvtColor(coin_blur, cv2.COLOR_BGR2GRAY)
plt.figure(), plt.imshow(coin_gray, cmap="gray"), plt.axis("off")
# Bulanıklaştırılmış görüntüyü tekrar gri tonlamaya çevirir.

# İkili eşikleme
ret, coin_thresh = cv2.threshold(coin_gray, 75, 255, cv2.THRESH_BINARY)
plt.figure(), plt.imshow(coin_thresh, cmap="gray"), plt.axis("off")
# Görüntüyü tekrar ikili eşikleme ile siyah-beyaz hale getirir.

# Açılma işlemi
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(coin_thresh, cv2.MORPH_OPEN, kernel, iterations=2)
plt.figure(), plt.imshow(opening, cmap="gray"), plt.axis("off")
# Açılma (morphological opening) işlemi ile küçük beyaz gürültüleri ve ince siyah çizgileri temizler.

# Nesneler arası distance (mesafe) bulma
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
plt.figure(), plt.imshow(dist_transform, cmap="gray"), plt.axis("off")
# Her bir pikselin en yakın beyaz piksele olan mesafesini hesaplar ve görüntüyü mesafe dönüşümü ile dönüştürür.

# Resmi küçültme (kesin ön plan belirleme)
ret, sure_foreground = cv2.threshold(dist_transform, 0.4 * np.max(dist_transform), 255, 0)
plt.figure(), plt.imshow(sure_foreground, cmap="gray"), plt.axis("off")
# Kesin ön planı belirlemek için mesafe dönüşümünü eşikleme işlemi ile ikili hale getirir.

# Arka plan için resmi büyütme
sure_background = cv2.dilate(opening, kernel, iterations=1)
sure_foreground = np.uint8(sure_foreground)
unknown = cv2.subtract(sure_background, sure_foreground)
plt.figure(), plt.imshow(unknown, cmap="gray"), plt.axis("off")
# Kesin arka planı belirlemek için dilate (genişletme) işlemi uygular ve bilinmeyen bölgeyi belirler.

# Bağlantı bileşenleri
ret, marker = cv2.connectedComponents(sure_foreground)
marker = marker + 1
marker[unknown == 255] = 0
plt.figure(), plt.imshow(marker, cmap="gray"), plt.axis("off")
# Kesin ön plan üzerinde bağlantı bileşenlerini bulur ve işaretleyici (marker) matrisi oluşturur.

# Havza (Watershed) algoritması
marker = cv2.watershed(coin, marker)
plt.figure(), plt.imshow(marker, cmap="gray"), plt.axis("off")
# Watershed algoritmasını kullanarak işaretleyici matrisini günceller.

# Kontur bulma
contours, hierarchy = cv2.findContours(marker.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
# Güncellenmiş işaretleyici matrisi üzerinde konturları bulur.

# Her bir konturu çizin
for i in range(len(contours)):
    if hierarchy[0][i][3] == -1:  # En dış kontur
        cv2.drawContours(coin, contours, i, (255, 0, 0), 2)
plt.figure(), plt.imshow(coin), plt.axis("off")
# Dış konturları kırmızı renkte ve ince çizgilerle çizer.
