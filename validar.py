import sys, cv2, numpy as np
a, b = cv2.VideoCapture(sys.argv[1]), cv2.VideoCapture(sys.argv[2])
maes, maximo, n = [], 0, 0
while True:
    ok1, fa = a.read(); ok2, fb = b.read()
    if not (ok1 and ok2): break
    d = np.abs(fa[50:].astype(int) - fb[50:].astype(int))  # ignora la etiqueta de texto
    maes.append(d.mean()); maximo = max(maximo, d.max()); n += 1
print(f"Cuadros: {n} | Error medio: {np.mean(maes):.4f} | Error maximo: {maximo} (niveles de gris, 0-255)")
