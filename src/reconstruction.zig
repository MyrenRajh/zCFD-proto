const std = @import("std");

pub fn exPWLinScalar(phi: anytype, grad_phi: anytype, r_of: anytype) @TypeOf(phi) {
    return phi + grad_phi.dot(r_of);
}

pub fn exPWLinVector(phi: anytype, grad_phi: anytype, r_of: anytype) @TypeOf(phi) {
    return phi.add(grad_phi.transposeMulVec(r_of));
}

pub fn exPWLinLimScalar(phi: anytype, grad_phi: anytype, r_of: anytype, limiter: anytype) @TypeOf(phi) {
    return phi + grad_phi.dot(r_of) * limiter;
}

pub fn exPWLinLimVector(phi: anytype, grad_phi: anytype, r_of: anytype, limiter: anytype) @TypeOf(phi) {
    return phi.add(grad_phi.transposeMulVec(r_of).mulElem(limiter));
}

pub fn inWeightScalar(phi_1: anytype, phi_2: anytype, w: anytype) @TypeOf(phi_1) {
    return (1.0 - w) * phi_1 + (w) * phi_2;
}

pub fn inWeightVector(phi_1: anytype, phi_2: anytype, w: anytype) @TypeOf(phi_1) {
    return phi_1.scale(1.0 - w).add(phi_2.scale(w));
}

pub fn inWeightMatrix(phi_1: anytype, phi_2: anytype, w: anytype) @TypeOf(phi_1) {
    return phi_1.scale(1.0 - w).add(phi_2.scale(w));
}

pub fn inPWLinScalar(
    phi_o: anytype,
    phi_n: anytype,
    grad_phi_o: anytype,
    grad_phi_n: anytype,
    r_of: anytype,
    r_nf: anytype,
) @TypeOf(phi_o) {
    const d_of = r_of.mag();
    const d_nf = r_nf.mag();
    const g_f = (d_of) / (d_of + d_nf);
    const phi_of = exPWLinScalar(phi_o, grad_phi_o, r_of);
    const phi_nf = exPWLinScalar(phi_n, grad_phi_n, r_nf);
    return inWeightScalar(phi_of, phi_nf, g_f);
}

pub fn inPWLinVector(
    phi_o: anytype,
    phi_n: anytype,
    grad_phi_o: anytype,
    grad_phi_n: anytype,
    r_of: anytype,
    r_nf: anytype,
) @TypeOf(phi_o) {
    const d_of = r_of.mag();
    const d_nf = r_nf.mag();
    const g_f = (d_of) / (d_of + d_nf);
    const phi_of = exPWLinVector(phi_o, grad_phi_o, r_of);
    const phi_nf = exPWLinVector(phi_n, grad_phi_n, r_nf);
    return inWeightVector(phi_of, phi_nf, g_f);
}

pub fn inLinGradScalar(
    phi_o: anytype,
    phi_n: anytype,
    grad_phi_o: anytype,
    grad_phi_n: anytype,
    r_of: anytype,
    r_nf: anytype,
    r_on: anytype,
) @TypeOf(grad_phi_o) {
    const d_of = r_of.mag();
    const d_nf = r_nf.mag();
    const d_on = r_on.mag();
    const e_on = r_on.unit();
    const g_f = (d_of) / (d_of + d_nf);
    const grad_phi_bar = inWeightVector(grad_phi_o, grad_phi_n, g_f);

    return grad_phi_bar.add(
        e_on.scale((phi_n - phi_o) / d_on),
    ).sub(e_on.scale(grad_phi_bar.dot(e_on)));
}

pub fn inLinGradVector(
    phi_o: anytype,
    phi_n: anytype,
    grad_phi_o: anytype,
    grad_phi_n: anytype,
    r_of: anytype,
    r_nf: anytype,
    r_on: anytype,
) @TypeOf(grad_phi_o) {
    const d_of = r_of.mag();
    const d_nf = r_nf.mag();
    const d_on = r_on.mag();
    const e_on = r_on.unit();
    const g_f = (d_of) / (d_of + d_nf);
    const grad_phi_bar = inWeightMatrix(grad_phi_o, grad_phi_n, g_f);

    return grad_phi_bar.add(
        e_on.outer(phi_n.sub(phi_o).scale(1.0 / d_on)),
    ).sub(e_on.outer(grad_phi_bar.transposeMulVec(e_on)));
}
