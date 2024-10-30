const std = @import("std");

pub fn PerfectGas(
    DType: anytype,
    // comptime constant_mu: bool,
    // comptime constant_k: bool,
) type {
    return struct {
        R: DType = 287.052874,
        gamma: DType = 1.4,
        Pr: DType = 0.72,
        c_p: DType = 1004.685059,
        c_v: DType = 717.632185,

        reference_p: DType = 101325,
        reference_v: DType = 350,
        reference_T: DType = 298,

        reference_mu: DType = 18.1e-6,
        reference_k: DType = 32.3e-3,

        // Ideal Gas Laws
        pub fn pressure(self: *@This(), rho: anytype, T: anytype) DType {
            return rho * self.R * T;
        }

        pub fn density(self: *@This(), p: anytype, T: anytype) DType {
            return p / (self.R * T);
        }

        pub fn temperature(self: *@This(), p: anytype, rho: anytype) DType {
            return p / (rho * self.R);
        }

        // Energy and Enthalpy
        pub fn energySpecific(self: *@This(), T: anytype) DType {
            return self.c_v * T;
        }

        pub fn enthalpySpecific(self: *@This(), T: anytype) DType {
            return self.c_p * T;
        }

        pub fn energyTotal(self: *@This(), T: anytype, v: anytype) DType {
            return energySpecific(self, T) + 0.5 * v.dot(v);
        }

        pub fn enthalpyTotal(self: *@This(), T: anytype, v: anytype) DType {
            return enthalpySpecific(self, T) + 0.5 * v.dot(v);
        }

        // Energy Conversions
        pub fn temperatureEnergyTotal(self: *@This(), E: anytype, v: anytype) DType {
            return (E - 0.5 * v.dot(v)) / self.c_v;
        }

        pub fn temperatureEnthalpyTotal(self: *@This(), H: anytype, v: anytype) DType {
            return (H - 0.5 * v.dot(v)) / self.c_p;
        }

        // Conversions
        pub fn gasConstantSpecific(self: *@This()) DType {
            return self.c_p - self.c_v;
        }

        pub fn heatCapacityRatio(self: *@This()) DType {
            return self.c_p / self.c_v;
        }

        pub fn heatCapacityPressure(self: *@This()) @TypeOf(DType) {
            return (self.gamma * self.R) / (self.gamma - 1.0);
        }

        pub fn heatCapacityTemperature(self: *@This()) DType {
            return self.R / (self.gamma - 1.0);
        }

        // // Sutherland Law Nondimensional
        // pub fn dynamicViscosity(self: *@This(), T: anytype) DType {
        //     const ratio_ref_T = 110.4 / self.reference_T;
        //     return self.reference_mu * std.math.pow(
        //         DType,
        //         T,
        //         1.5,
        //     ) * (1.0 + ratio_ref_T) / (T + ratio_ref_T);
        // }

        // Sutherland Law Nondimensional
        pub fn dynamicViscosity(self: *@This(), T: anytype) DType {
            return self.reference_mu * std.math.pow(
                DType,
                T / self.reference_T,
                1.5,
            ) * (self.reference_T + 110.4) / (T + 110.4);
        }

        pub fn thermalConductivity(self: *@This(), mu: anytype) DType {
            return self.c_p * mu / self.Pr;
        }

        // Primitive from Conservative
        pub fn pressureConservative(self: *@This(), rho: anytype, rho_v: anytype, rho_E: anytype) DType {
            return (self.gamma - 1.0) * (rho_E - rho_v.dot(rho_v) / (2.0 * rho));
        }

        // Speed of Sound
        pub fn soundSpeed(self: *@This(), T: anytype) DType {
            return @sqrt(self.gamma * self.R * T);
        }

        pub fn soundSpeedEnthalpy(self: *@This(), H: anytype, v: anytype) @TypeOf(H) {
            // _ = self;
            return @sqrt((self.gamma - 1.0) * (H - 0.5 * v.dot(v)));
        }

        pub fn machNumber(self: *@This(), c: anytype, v: anytype) @TypeOf(c) {
            _ = self;
            return v.mag() / c;
        }

        // Stress - Strain - Newtonion Fluid Stokes Hypothesis
        pub fn viscousStress(self: *@This(), mu: anytype, grad_v: anytype) @TypeOf(grad_v) {
            _ = self;
            const MatrixType = @TypeOf(grad_v);
            const div_v = grad_v.trace();
            const sum_v_vt = grad_v.add(grad_v.transpose());
            const diag_div_v = MatrixType.diagSplat(div_v * 2.0 / 3.0);
            return sum_v_vt.sub(diag_div_v).scale(mu);
        }
    };
}
