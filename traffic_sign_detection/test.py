


# gray_img = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
# dx, dy = np.gradient(gray_img)
#
# dx = dx.astype(np.float)
# dy = dy.astype(np.float)
#
# d_I = dx + dy
#
# cv2.imwrite("dx.png", dx)
# cv2.imwrite("dy.png", dy)
# cv2.imwrite("d_I.png", d_I)

# cv2.imwrite("out/edges.png", edges)
# edges = cv2.imread("out/edges.png", cv2.IMREAD_GRAYSCALE)
map = {}

map['list'] = 1

l = map.get('list')

print(l)

s = {0, 1, 2} # set([0, 1, 2])

s.remove(2)
s.remove(0)
print(s.pop())
