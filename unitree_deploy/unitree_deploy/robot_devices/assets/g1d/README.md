# Unitree G1-D model

`g1_d.urdf`, `meshes/` and `LICENSE` (BSD-3-Clause) are copied unchanged from
Unitree's `unitree_ros` repository, revision `2fc4f3087920f4c309676e5389f6ede13e2448a8`,
`robots/g1_d_description/`, the same files used by the DGS G1-D Cyber Twin
(`public/model/provenance.json` there lists their SHA-256).

The meshes of the three-finger hands (`*hand*.STL`) are not included:
`scripts/sim_g1_robot.py --robot g1d` removes the hands from the URDF when
loading it and mounts Dex1-style two-finger grippers instead.
