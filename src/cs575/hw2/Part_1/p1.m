[X, MAP] = imread('p1_search.png');
imwrite(X,MAP,'p1_search.jpg','jpg');
a = imread('p1_search.jpg');
bw = im2bw(a, 0.9);
bi = ~bw;
se_v = strel('line',50,0);
se_v = strel('line',50,90);
se_h = strel('line',50,0);
e1 = imerode(bi, se_v);
d1 = imdilate(e1, se_v);
e2 = imerode(bi, se_h);
d2 = imdilate(e2, se_h);
grid = bitor(d1, d2);
im_fill = imfill(grid, 'holes');
s = regionprops(im_fill,'BoundingBox');
out = imcrop(a, [s.BoundingBox(1),s.BoundingBox(2),s.BoundingBox(3),s.BoundingBox(4)]);
imshow(out)
imwrite(out,'out.jpg','jpg')