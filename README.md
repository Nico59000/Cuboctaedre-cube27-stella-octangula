# Cuboctaedre-cube27-stella-octangula
petit script de validation de formulation obtenues : 

Pour le lancer en manipulation interactive :

python cube27_cubocta_stella_render.py

Pour un MP4 :

python cube27_cubocta_stella_render.py \
  --save cube27_cubocta_stella.mp4 \
  --frames 240 --fps 30

Et avec une rotation complète de caméra pendant l’alternance :

python cube27_cubocta_stella_render.py \
  --save cube27_cubocta_stella_rotating.mp4 \
  --frames 240 --fps 30 --rotate-camera


La version VTK est presque prête et requière : 

pip install numpy vtk

