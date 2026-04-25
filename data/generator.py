import sympy as sp
import random
import os

x = sp.Symbol('x')


def format_poly(expr):
    s = str(sp.Poly(expr, x).as_expr())
    s = s.replace('**', '^').replace('*', '').replace(' ', '')
    s = s.replace('(', '').replace(')', '')
    return s


def generate_sample(max_degree, coeff_range):
    degree = random.randint(1, max_degree)

    coeffs = [random.randint(coeff_range[0], coeff_range[1]) for _ in range(degree + 1)]
    while coeffs[0] == 0:
        coeffs[0] = random.randint(coeff_range[0], coeff_range[1])

    expr = sum(c * x**(degree - i) for i, c in enumerate(coeffs))
    deriv = sp.diff(expr, x)

    input_str = format_poly(expr)

    if deriv == 0:
        output_str = "0"
    else:
        output_str = format_poly(deriv)

    return input_str, output_str


def save_split(data, directory):
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, 'inputs.txt'), 'w') as fi:
        for inp, _ in data:
            fi.write(inp + '\n')
    with open(os.path.join(directory, 'outputs.txt'), 'w') as fo:
        for _, out in data:
            fo.write(out + '\n')


def main():
    random.seed(42)

    train_data = []

    for _ in range(40000):
        train_data.append(generate_sample(2, (-5, 5)))
    for _ in range(60000):
        train_data.append(generate_sample(3, (-5, 5)))

    random.shuffle(train_data)
    save_split(train_data, 'data/train')

    random.seed(99)
    val_data = [generate_sample(3, (-5, 5)) for _ in range(2000)]
    save_split(val_data, 'data/val')

    random.seed(13)
    ood_data = [generate_sample(3, (-8, 8)) for _ in range(500)]
    save_split(ood_data, 'data/ood')

    print(f"train: {len(train_data)}")
    print(f"val: {len(val_data)}")
    print(f"ood: {len(ood_data)}")

    vocab = set('x+-^0123456789')
    for i in range(20):
        inp, out = train_data[i]
        bad_chars_inp = [c for c in inp if c not in vocab]
        bad_chars_out = [c for c in out if c not in vocab]
        if bad_chars_inp or bad_chars_out:
            print(f"BAD CHARS: input={inp} output={out} bad={bad_chars_inp + bad_chars_out}")

    print("\nsamples:")
    for i in range(10):
        inp, out = train_data[i]
        print(f"  {inp}  ->  {out}")


main()