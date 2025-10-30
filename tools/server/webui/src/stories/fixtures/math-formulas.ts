// Math Formulas Content
export const MATH_FORMULAS_MD = String.raw`
# Mathematical Formulas and Expressions

This document demonstrates various mathematical notation and formulas that can be rendered using LaTeX syntax in markdown.

## Basic Arithmetic

### Addition and Summation
$$\sum_{i=1}^{n} i = \frac{n(n+1)}{2}$$

## Algebra

### Quadratic Formula
The solutions to $ax^2 + bx + c = 0$ are:
$$x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}$$

### Binomial Theorem
$$(x + y)^n = \sum_{k=0}^{n} \binom{n}{k} x^{n-k} y^k$$

## Calculus

### Derivatives
The derivative of $f(x) = x^n$ is:
$$f'(x) = nx^{n-1}$$

### Integration
$$\int_a^b f(x) \, dx = F(b) - F(a)$$

### Fundamental Theorem of Calculus
$$\frac{d}{dx} \int_a^x f(t) \, dt = f(x)$$

## Linear Algebra

### Matrix Multiplication
If $A$ is an $m \times n$ matrix and $B$ is an $n \times p$ matrix, then:
$$C_{ij} = \sum_{k=1}^{n} A_{ik} B_{kj}$$

### Eigenvalues and Eigenvectors
For a square matrix $A$, if $Av = \lambda v$ for some non-zero vector $v$, then:
- $\lambda$ is an eigenvalue
- $v$ is an eigenvector

## Statistics and Probability

### Normal Distribution
The probability density function is:
$$f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2}$$

### Bayes' Theorem
$$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$$

### Central Limit Theorem
For large $n$, the sample mean $\bar{X}$ is approximately:
$$\bar{X} \sim N\left(\mu, \frac{\sigma^2}{n}\right)$$

## Trigonometry

### Pythagorean Identity
$$\sin^2\theta + \cos^2\theta = 1$$

### Euler's Formula
$$e^{i\theta} = \cos\theta + i\sin\theta$$

### Taylor Series for Sine
$$\sin x = \sum_{n=0}^{\infty} \frac{(-1)^n}{(2n+1)!} x^{2n+1} = x - \frac{x^3}{3!} + \frac{x^5}{5!} - \frac{x^7}{7!} + \cdots$$

## Complex Analysis

### Complex Numbers
A complex number can be written as:
$$z = a + bi = r e^{i\theta}$$

where $r = |z| = \sqrt{a^2 + b^2}$ and $\theta = \arg(z)$

### Cauchy-Riemann Equations
For a function $f(z) = u(x,y) + iv(x,y)$ to be analytic:
$$\frac{\partial u}{\partial x} = \frac{\partial v}{\partial y}, \quad \frac{\partial u}{\partial y} = -\frac{\partial v}{\partial x}$$

## Differential Equations

### First-order Linear ODE
$$\frac{dy}{dx} + P(x)y = Q(x)$$

Solution: $y = e^{-\int P(x)dx}\left[\int Q(x)e^{\int P(x)dx}dx + C\right]$

### Heat Equation
$$\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}$$

## Number Theory

### Prime Number Theorem
$$\pi(x) \sim \frac{x}{\ln x}$$

where $\pi(x)$ is the number of primes less than or equal to $x$.

### Fermat's Last Theorem
For $n > 2$, there are no positive integers $a$, $b$, and $c$ such that:
$$a^n + b^n = c^n$$

## Set Theory

### De Morgan's Laws
$$\overline{A \cup B} = \overline{A} \cap \overline{B}$$
$$\overline{A \cap B} = \overline{A} \cup \overline{B}$$

## Advanced Topics

### Riemann Zeta Function
$$\zeta(s) = \sum_{n=1}^{\infty} \frac{1}{n^s} = \prod_{p \text{ prime}} \frac{1}{1-p^{-s}}$$

### Maxwell's Equations
$$\nabla \cdot \mathbf{E} = \frac{\rho}{\epsilon_0}$$
$$\nabla \cdot \mathbf{B} = 0$$
$$\nabla \times \mathbf{E} = -\frac{\partial \mathbf{B}}{\partial t}$$
$$\nabla \times \mathbf{B} = \mu_0\mathbf{J} + \mu_0\epsilon_0\frac{\partial \mathbf{E}}{\partial t}$$

### Schrödinger Equation
$$i\hbar\frac{\partial}{\partial t}\Psi(\mathbf{r},t) = \hat{H}\Psi(\mathbf{r},t)$$

## Inline Math Examples

Here are some inline mathematical expressions:

- The golden ratio: $\phi = \frac{1 + \sqrt{5}}{2} \approx 1.618$
- Euler's number: $e = \lim_{n \to \infty} \left(1 + \frac{1}{n}\right)^n$
- Pi: $\pi = 4 \sum_{n=0}^{\infty} \frac{(-1)^n}{2n+1}$
- Square root of 2: $\sqrt{2} = 1.41421356...$

## Fractions and Radicals

Complex fraction: $\frac{\frac{a}{b} + \frac{c}{d}}{\frac{e}{f} - \frac{g}{h}}$

Nested radicals: $\sqrt{2 + \sqrt{3 + \sqrt{4 + \sqrt{5}}}}$

## Summations and Products

### Geometric Series
$$\sum_{n=0}^{\infty} ar^n = \frac{a}{1-r} \quad \text{for } |r| < 1$$

### Product Notation
$$n! = \prod_{k=1}^{n} k$$

### Double Summation
$$\sum_{i=1}^{m} \sum_{j=1}^{n} a_{ij}$$

## Limits

$$\lim_{x \to 0} \frac{\sin x}{x} = 1$$

$$\lim_{n \to \infty} \left(1 + \frac{x}{n}\right)^n = e^x$$

---

*This document showcases various mathematical notation and formulas that can be rendered in markdown using LaTeX syntax.*
`;
