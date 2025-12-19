# Modified copy of https://github.com/elerac/sh_table Spherical Harmonics Table Generator.
# Used to generate a function evaluating spherical harmonics up to an arbitrary degree.
# Simplified as much as possible (constant parts pre-evaluated and no pow(...) calls).

# Generate a table of spherical harmonics
from pathlib import Path
from sympy import *
from sympy.codegen.ast import real, float32

def P(l, m, x):
    """https://en.wikipedia.org/wiki/Associated_Legendre_polynomials"""
    if m == 0:
        return legendre(l, x)
    elif m > 0:
        return (-1) ** m * sqrt(1 - x**2) ** m * diff(legendre(l, x), x, m)
    else:
        m = -m
        # Note the relation with m > 0 case.
        return (-1) ** m * factorial(l - m) / factorial(l + m) * P(l, m, x)

def Y(l, m, theta, phi):
    """https://en.wikipedia.org/wiki/Spherical_harmonics#Orthogonality_and_normalization"""

    cos_theta = symbols("cos_theta", real=True)  # temporary variable for cos(theta)
    Y_l_m = sqrt((2 * l + 1) / (4 * pi) * factorial(l - m) / factorial(l + m)) * P(l, m, cos_theta) * exp(I * m * phi)

    Y_l_m = Y_l_m.subs(cos_theta, cos(theta))

    return Y_l_m

def Y_real(l, m, theta, phi):
    """https://en.wikipedia.org/wiki/Spherical_harmonics#Real_form"""
    if m == 0:
        return Y(l, m, theta, phi)
    elif m > 0:
        return (Y(l, -m, theta, phi) + (-1) ** m * Y(l, m, theta, phi)) / sqrt(2)
    else:
        return I * (Y(l, m, theta, phi) - (-1) ** m * Y(l, -m, theta, phi)) / sqrt(2)

# Replace pow by mul.
def sack(expr):
    return expr.replace(
        lambda x: x.is_Pow and x.exp > 0,
        lambda x: Symbol('*'.join([x.base.name]*x.exp)))

def main():
    # Maximum level
    levels = 5

    # symbols
    x, y, z = symbols("x y z", real=True)
    r = symbols("r", real=True, positive=True)
    theta, phi = symbols("theta phi", real=True)

    expressions = {}
    for l in range(levels):
        for m in range(-l, l + 1):
            Y_l_m_real = Y_real(l, m, theta, phi)

            # Replace [theta, phi] with [x, y, z]
            # exp(I * phi) = (x + I * y) / sqrt(x**2 + y**2)
            # phi = atan2(y, x)
            # theta = acos(z / r)
            Y_l_m_real = Y_l_m_real.subs(exp(I * phi), (x + I * y) / sqrt(x**2 + y**2))
            Y_l_m_real = Y_l_m_real.subs(phi, atan2(y, x))
            Y_l_m_real = Y_l_m_real.subs(theta, acos(z / sqrt(x**2 + y**2 + z**2)))

            # Replace x**2 + y**2 + z**2 with r**2
            Y_l_m_real = Y_l_m_real.subs(x**2 + y**2 + z**2, r**2).factor()
            Y_l_m_real = Y_l_m_real.subs(x**2, r**2 - y**2 - z**2).factor()
            Y_l_m_real = Y_l_m_real.subs(y**2, r**2 - x**2 - z**2).factor()
            Y_l_m_real = Y_l_m_real.subs(z**2, r**2 - x**2 - y**2).factor()

            # Get real part
            Y_l_m_real = re(Y_l_m_real)

            Y_l_m_real = Y_l_m_real.simplify()

            expressions[(l, m)] = Y_l_m_real

            if m >= 0:
                print(f"l={l}, m={m}  -> {Y_l_m_real}")
            else:
                print(f"l={l}, m={m} -> {Y_l_m_real}")

    def access_buffer_sh():
        sh_counter = 0
        while True:
            yield 'loadReadOnly(&sh[{} * stride])'.format(sh_counter)
            sh_counter = sh_counter + 1

    def read_sh_vec3():
        local_sh_counter = 0
        gen_buffer_sh = access_buffer_sh()
        while True:
            local_var = 'sh{}'.format(local_sh_counter)
            local_assign = 'auto {} = glm::vec3({}, {}, {});\n'.format(local_var, next(gen_buffer_sh), next(gen_buffer_sh), next(gen_buffer_sh))
            yield local_assign, local_var
            local_sh_counter = local_sh_counter + 1

    gen_sh_var = read_sh_vec3()

    xx, yy, zz = symbols("xx yy zz", real=True, positive=True)

    indent = 0

    def indt(indent):
        return ' ' * 4 * indent;

    code_str = '// clang-format off\n'
    code_str += '__device__ glm::vec3 sphericalHarmonics(const int l, const glm::vec3& dir, const float* sh, const int stride)\n'
    code_str += '{\n'

    indent = indent + 1

    code_str += indt(indent) + 'const auto x = dir.x;\n'
    code_str += indt(indent) + 'const auto y = dir.y;\n'
    code_str += indt(indent) + 'const auto z = dir.z;\n'
    code_str += indt(indent) + 'auto result = glm::vec3(0.0f);\n'

    for l in range(0, levels):

        code_str += '\n'
        code_str += indt(indent) + '// Level {}.\n'.format(l)

        if l > 0:
            code_str += indt(indent) + 'if (l > {})\n'.format(l - 1)
            code_str += indt(indent) + '{\n'
            indent = indent + 1

        if l == 2:
            code_str += indt(indent) + 'auto xx = x * x;\n'
            code_str += indt(indent) + 'auto yy = y * y;\n'
            code_str += indt(indent) + 'auto zz = z * z;\n'
            code_str += '\n'

        local_vars = list()

        # Local variables.
        for m in range(-l, l + 1):
            local_assign, local_var = next(gen_sh_var)
            code_str += indt(indent) + local_assign
            local_vars.append(local_var)

        code_str += '\n'
        local_var_idx = 0

        code_str += indt(indent) + 'result += \n'
        indent = indent + 1
        for m in range(-l, l + 1):
            Y_l_m_real = expressions[(l, m)]

            # Use xx, yy, zz
            #Y_l_m_real = Y_l_m_real.expand()
            Y_l_m_real = Y_l_m_real.subs(x*x, xx)
            Y_l_m_real = Y_l_m_real.subs(y*y, yy)
            Y_l_m_real = Y_l_m_real.subs(z*z, zz)

            # r = 1
            Y_l_m_real = Y_l_m_real.subs(r, 1)

            Y_l_m_real = Y_l_m_real.simplify()

            # Evaluate sqrt values
            Y_l_m_real = Y_l_m_real.evalf()

            # Replace pow by mul.
            Y_l_m_real = sack(Y_l_m_real)
            
            # Replace to new one
            expressions[(l, m)] = Y_l_m_real

            compute_str = ccode(Y_l_m_real, standard='C99', type_aliases={real: float32})
            # Apply coeffcient.
            compute_str = '({}) * {}'.format(compute_str, local_vars[local_var_idx])
            if m == l:
                code_str += indt(indent) + '{};\n'.format(compute_str)
            else:
                code_str += indt(indent) + '{} +\n'.format(compute_str)

            local_var_idx = local_var_idx + 1
        indent = indent - 1
        assert(local_var_idx == len(local_vars))

    for l in range(1, levels):
        indent = indent - 1
        code_str += indt(indent) + '}\n' # End if l > ...

    code_str += indt(indent) + 'return glm::clamp(result + 0.5f);\n'
    indent = indent - 1
    code_str += indt(indent) + '}\n'
    code_str += '// clang-format on\n'
    print(code_str)

if __name__ == "__main__":
    main()
