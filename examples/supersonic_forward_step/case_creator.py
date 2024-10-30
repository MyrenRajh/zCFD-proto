import numpy as np
import openfoamparser as ofp
import os as os
import shutil as shutil
import enum as enum


class TimeIntegrationScheme(enum.Enum):
    exp_euler = 0
    exp_multi_rk_1_3 = 1
    exp_multi_rk_1_4 = 2
    exp_multi_rk_1_5 = 3
    exp_multi_rk_2_3 = 4
    exp_multi_rk_2_4 = 5
    exp_multi_rk_2_5 = 6


class LimiterType(enum.Enum):
    venkatakrishnan = 0
    barth_jeperson = 1


class ConvectiveFluxScheme(enum.Enum):
    none = 0
    riemann_roe = 1
    riemann_hllc = 2


class ViscousFluxScheme(enum.Enum):
    none = 0
    interp_grad_linear = 1


def initCase(path):
    new_path = os.path.join(path, "case")

    if os.path.exists(new_path):
        shutil.rmtree(new_path)

    os.mkdir(new_path)
    os.mkdir(os.path.join(new_path, "mesh"))
    os.mkdir(os.path.join(new_path, "controls"))
    os.mkdir(os.path.join(new_path, "solution"))


def readOpenFoamMesh(case_path):
    of_mesh = ofp.FoamMesh(case_path)

    node_centroids = of_mesh.points

    face_i_owner = np.array(of_mesh.owner)
    face_i_neighbour = np.array(of_mesh.neighbour)

    face_i_nodes = []
    face_n_nodes = []

    for i in range(len(of_mesh.faces)):
        face_i_nodes += of_mesh.faces[i]
        face_n_nodes.append(len(of_mesh.faces[i]))

    face_i_nodes = np.array(face_i_nodes, dtype=np.int64)
    face_n_nodes = np.array(face_n_nodes, dtype=np.int64)
    face_i_start_node = np.hstack([np.zeros(1, dtype=np.int64), face_n_nodes]).cumsum()[
        0:-1
    ]

    face_i_owner = np.array(of_mesh.owner, dtype=np.int64)
    face_i_neighbour = np.array(of_mesh.neighbour, dtype=np.int64)

    cell_i_faces = []
    cell_n_faces = []

    for i in range(len(of_mesh.cell_faces)):
        cell_i_faces += of_mesh.cell_faces[i]
        cell_n_faces.append(len(of_mesh.cell_faces[i]))

    cell_i_faces = np.array(cell_i_faces, dtype=np.int64)
    cell_n_faces = np.array(cell_n_faces, dtype=np.int64)
    cell_i_start_face = np.hstack([np.zeros(1, dtype=np.int64), cell_n_faces]).cumsum()[
        0:-1
    ]

    face_i_neighbour[face_i_neighbour < 0] = 0

    boundary_i_start_face = np.zeros(len(of_mesh.boundary), dtype=np.int64)
    boundary_n_faces = np.zeros(len(of_mesh.boundary), dtype=np.int64)

    for i in range(len(of_mesh.boundary)):
        boundary_i_start_face[i] = of_mesh.boundary[
            list(of_mesh.boundary.keys())[i]
        ].start
        boundary_n_faces[i] = of_mesh.boundary[list(of_mesh.boundary.keys())[i]].num

    N_faces = len(of_mesh.faces)
    N_nodes = of_mesh.points.shape[0]
    N_cells = len(of_mesh.cell_faces)
    N_boundary_patches = len(of_mesh.boundary)

    N_faces_internal = of_mesh.num_inner_face
    N_faces_external = N_faces - N_faces_internal

    # Dummy Cells
    N_cells_dummy = N_faces_external

    cell_n_faces = np.hstack([cell_n_faces, np.ones(N_cells_dummy, dtype=np.int64)])
    cell_i_start_face = np.hstack(
        [
            cell_i_start_face,
            np.arange(cell_i_faces.shape[0], cell_i_faces.shape[0] + N_cells_dummy),
        ]
    )
    cell_i_faces = np.hstack([cell_i_faces, np.arange(N_faces_internal, N_faces)])

    mesh_info = np.array(
        [
            N_nodes,
            N_faces,
            N_cells,
            N_boundary_patches,
            N_faces_internal,
            N_faces_external,
            N_cells_dummy,
        ]
    )

    return {
        # Mesh Info
        "mesh_info": mesh_info,
        # Node Info
        "node_centroid": node_centroids,
        # Face Info
        "face_i_owner": face_i_owner,
        "face_i_neighbour": face_i_neighbour,
        "face_n_nodes": face_n_nodes,
        "face_i_start_node": face_i_start_node,
        # Cell Info
        "cell_n_faces": cell_n_faces,
        "cell_i_start_face": cell_i_start_face,
        # Boundary Info
        "boundary_i_start_face": boundary_i_start_face,
        "boundary_n_faces": boundary_n_faces,
        # Connectivity Info
        "face_i_nodes": face_i_nodes,
        "cell_i_faces": cell_i_faces,
    }


def writeOpenFoamMesh(path, mesh):
    mesh_path = os.path.join(path, "case/mesh")

    if os.path.exists(mesh_path):
        shutil.rmtree(mesh_path)

    node_path = os.path.join(mesh_path, "node")
    face_path = os.path.join(mesh_path, "face")
    cell_path = os.path.join(mesh_path, "cell")
    boundary_path = os.path.join(mesh_path, "boundary")

    os.mkdir(mesh_path)
    os.mkdir(node_path)
    os.mkdir(face_path)
    os.mkdir(cell_path)
    os.mkdir(boundary_path)

    np.savetxt(
        os.path.join(mesh_path, "info"), mesh["mesh_info"], fmt="%d", delimiter="\t"
    )

    np.savetxt(
        os.path.join(node_path, "centroid"),
        mesh["node_centroid"],
        fmt="%.8e",
        delimiter="\t",
    )

    np.savetxt(
        os.path.join(face_path, "i_owner"),
        mesh["face_i_owner"],
        fmt="%d",
        delimiter="\t",
    )
    np.savetxt(
        os.path.join(face_path, "i_neighbour"),
        mesh["face_i_neighbour"],
        fmt="%d",
        delimiter="\t",
    )
    np.savetxt(
        os.path.join(face_path, "n_nodes"),
        mesh["face_n_nodes"],
        fmt="%d",
        delimiter="\t",
    )
    np.savetxt(
        os.path.join(face_path, "i_start_node"),
        mesh["face_i_start_node"],
        fmt="%d",
        delimiter="\t",
    )
    np.savetxt(
        os.path.join(face_path, "i_nodes"),
        mesh["face_i_nodes"],
        fmt="%d",
        delimiter="\t",
    )

    np.savetxt(
        os.path.join(cell_path, "n_faces"),
        mesh["cell_n_faces"],
        fmt="%d",
        delimiter="\t",
    )
    np.savetxt(
        os.path.join(cell_path, "i_start_face"),
        mesh["cell_i_start_face"],
        fmt="%d",
        delimiter="\t",
    )
    np.savetxt(
        os.path.join(cell_path, "i_faces"),
        mesh["cell_i_faces"],
        fmt="%d",
        delimiter="\t",
    )

    np.savetxt(
        os.path.join(boundary_path, "i_start_face"),
        mesh["boundary_i_start_face"],
        fmt="%d",
        delimiter="\t",
    )
    np.savetxt(
        os.path.join(boundary_path, "n_faces"),
        mesh["boundary_n_faces"],
        fmt="%d",
        delimiter="\t",
    )


class BCType(enum.Enum):
    symmetry = 0

    wall_viscous_heated = 1
    wall_inviscid_heated = 2

    wall_viscous_isothermal = 3
    wall_inviscid_isothermal = 4

    inlet_supersonic = 5
    inlet_subsonic_velocity = 6
    inlet_subsonic_static_pressure = 7
    inlet_subsonic_total_pressure = 8

    outlet_supersonic = 9
    outlet_subsonic_pressure = 10
    outlet_subsonic_mass_flow = 11
    outlet_subsonic_developed = 12

    farfield_subsonic_inflow = 13
    farfield_subsonic_outflow = 14

    farfield_supersonic_inflow = 15
    farfield_supersonic_outflow = 16


def writeBoundaryConditions(
    path,
    bc_types,
    bc_p_values,
    bc_v_values,
    bc_T_values,
    bc_grad_p_normal_values,
    bc_grad_v_normal_values,
    bc_grad_T_normal_values,
):
    bc_controls_path = os.path.join(path, "case/controls/boundary")
    if os.path.exists(bc_controls_path):
        shutil.rmtree(bc_controls_path)
    os.mkdir(bc_controls_path)

    bc_types_arr = np.array([i.value for i in bc_types], dtype=np.int64)

    np.savetxt(
        os.path.join(bc_controls_path, "type"),
        bc_types_arr,
        fmt="%d",
        delimiter="\t",
    )

    np.savetxt(
        os.path.join(bc_controls_path, "p"),
        bc_p_values,
        fmt="%.8e",
        delimiter="\t",
    )

    np.savetxt(
        os.path.join(bc_controls_path, "v"),
        bc_v_values,
        fmt="%.8e",
        delimiter="\t",
    )

    np.savetxt(
        os.path.join(bc_controls_path, "T"),
        bc_T_values,
        fmt="%.8e",
        delimiter="\t",
    )

    np.savetxt(
        os.path.join(bc_controls_path, "grad_p_normal"),
        bc_grad_p_normal_values,
        fmt="%.8e",
        delimiter="\t",
    )

    np.savetxt(
        os.path.join(bc_controls_path, "grad_v_normal"),
        bc_grad_v_normal_values,
        fmt="%.8e",
        delimiter="\t",
    )

    np.savetxt(
        os.path.join(bc_controls_path, "grad_T_normal"),
        bc_grad_T_normal_values,
        fmt="%.8e",
        delimiter="\t",
    )


def writeInitialConditions(
    path,
    mesh,
    p_init,
    v_init,
    T_init,
):
    solution_path = os.path.join(path, "case/solution/0")
    if os.path.exists(solution_path):
        shutil.rmtree(solution_path)
    os.mkdir(solution_path)

    ones = np.ones(mesh["mesh_info"][2], dtype=np.float64)
    ones_vec = np.ones((mesh["mesh_info"][2], 3), dtype=np.float64)

    np.savetxt(
        os.path.join(solution_path, "p"),
        ones * p_init,
        fmt="%.8e",
        delimiter="\t",
    )

    np.savetxt(
        os.path.join(solution_path, "v"),
        ones_vec * v_init,
        fmt="%.8e",
        delimiter="\t",
    )

    np.savetxt(
        os.path.join(solution_path, "T"),
        ones * T_init,
        fmt="%.8e",
        delimiter="\t",
    )


def writeFluidProperties(
    path,
    R=287.0,
    gamma=1.4,
    Pr=0.72,
    c_p=1004.0,
    c_v=717.6,
    reference_p=101325.0,
    reference_v=350.0,
    reference_T=298.0,
    reference_mu=18.1e-6,
    reference_k=32.3e-3,
):
    controls_path = os.path.join(path, "case/controls")

    props = np.array(
        [
            R,
            gamma,
            Pr,
            c_p,
            c_v,
            reference_p,
            reference_v,
            reference_T,
            reference_mu,
            reference_k,
        ],
        dtype=np.float64,
    )

    np.savetxt(
        os.path.join(controls_path, "properties"),
        props,
        fmt="%.8e",
        delimiter="\t",
    )


def writeSystemControls(
    path,
    n_threads=4,
    save_every_n_steps=1,
    n_sim_steps=1000,
    time_integration_scheme=TimeIntegrationScheme.exp_euler,
    use_local_time_stepping=False,
    auto_calc_time_step=True,
    dt=1.0,
    cfl=0.5,
    local_dt_C_factor=2.0,
    limiter_type=LimiterType.venkatakrishnan,
    limiter_k_factor=1.0,
    convective_flux_scheme=ConvectiveFluxScheme.riemann_hllc,
    viscous_flux_scheme=ViscousFluxScheme.interp_grad_linear,
):
    controls_path = os.path.join(path, "case/controls")

    controls = np.array(
        [
            n_threads,
            save_every_n_steps,
            n_sim_steps,
            time_integration_scheme.value,
            int(use_local_time_stepping),
            int(auto_calc_time_step),
            limiter_type.value,
            convective_flux_scheme.value,
            viscous_flux_scheme.value,
        ],
        dtype=np.int32,
    )

    parameters = np.array(
        [
            dt,
            cfl,
            local_dt_C_factor,
            limiter_k_factor
        ],
        dtype=np.float64,
    )

    np.savetxt(
        os.path.join(controls_path, "system"),
        controls,
        fmt="%d",
        delimiter="\t",
    )

    np.savetxt(
        os.path.join(controls_path, "parameters"),
        parameters,
        fmt="%.8e",
        delimiter="\t",
    )
