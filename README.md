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


La version VTK plus musclée en besoins graphiques requière : 

pip install numpy vtk

Pour privilégier encore davantage la vitesse :

python cube27_cubocta_stella_render_vtk.py \
    --quality low \
    --msaa 0 \
    --target-fps 120

Pour une machine disposant d'un GPU confortable :

python cube27_cubocta_stella_render_vtk.py \
    --quality high \
    --msaa 4 \
    --target-fps 60


cube27_cubocta_stella_render_vtk_morph_bidirectional_R2.py est la version la plus récente prenant en compte les corrections droite et gauche de fixation à la grille 3d cube 27 des noeuds, corrections désactivables avec L et M si voulu . 
