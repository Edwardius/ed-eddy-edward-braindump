Here's a comprehensive cheatsheet of all your LaTeX shortcuts:

## Math Mode Triggers

| Trigger | Replacement                | Description           | Example                       |
| ------- | -------------------------- | --------------------- | ----------------------------- |
| `mk`    | `$$0$`                     | Inline math           | `$x^2$` → $x^2$               |
| `dm`    | `$$\n$0\n$$`               | Display math block    | `$$x^2$$` → $$x^2$$           |
| `beg`   | `\begin{$0}\n$1\n\end{$0}` | Begin/end environment | `\begin{align}...\end{align}` |

## Greek Letters

|Trigger|Replacement|Example|
|---|---|---|
|`@a`|`\alpha`|$\alpha$|
|`@b`|`\beta`|$\beta$|
|`@g`|`\gamma`|$\gamma$|
|`@G`|`\Gamma`|$\Gamma$|
|`@d`|`\delta`|$\delta$|
|`@D`|`\Delta`|$\Delta$|
|`@e`|`\epsilon`|$\epsilon$|
|`:e`|`\varepsilon`|$\varepsilon$|
|`@z`|`\zeta`|$\zeta$|
|`@t`|`\theta`|$\theta$|
|`@T`|`\Theta`|$\Theta$|
|`:t`|`\vartheta`|$\vartheta$|
|`@i`|`\iota`|$\iota$|
|`@k`|`\kappa`|$\kappa$|
|`@l`|`\lambda`|$\lambda$|
|`@L`|`\Lambda`|$\Lambda$|
|`@s`|`\sigma`|$\sigma$|
|`@S`|`\Sigma`|$\Sigma$|
|`@u`|`\upsilon`|$\upsilon$|
|`@U`|`\Upsilon`|$\Upsilon$|
|`@o`|`\omega`|$\omega$|
|`@O`|`\Omega`|$\Omega$|
|`ome`|`\omega`|$\omega$|
|`Ome`|`\Omega`|$\Omega$|

## Text in Math Mode

| Trigger | Replacement   | Description      | Example                         |
| ------- | ------------- | ---------------- | ------------------------------- |
| `text`  | `\text{$0}$1` | Text environment | `\text{hello}` → $\text{hello}$ |
| `"`     | `\text{$0}$1` | Quick text       | Same as above                   |
|         |               |                  |                                 |

## Basic Operations

|Trigger|Replacement|Description|Example|
|---|---|---|---|
|`sr`|`^{2}`|Square|`x sr` → $x^{2}$|
|`cb`|`^{3}`|Cube|`x cb` → $x^{3}$|
|`rd`|`^{$0}$1`|Power|`x rd` → $x^{n}$|
|`_`|`_{$0}$1`|Subscript|`x_` → $x_{i}$|
|`sts`|`_\text{$0}`|Text subscript|`x sts` → $x_\text{max}$|
|`sq`|`\sqrt{ $0 }$1`|Square root|`sq` → $\sqrt{x}$|
|`//`|`\frac{$0}{$1}$2`|Fraction|`//` → $\frac{a}{b}$|
|`ee`|`e^{ $0 }$1`|Exponential|`ee` → $e^{x}$|
|`invs`|`^{-1}`|Inverse|`A invs` → $A^{-1}$|
|`conj`|`^{*}`|Conjugate|`z conj` → $z^{*}$|

## Auto Subscripts

|Trigger|Replacement|Description|Example|
|---|---|---|---|
|`x2`|`x_{2}`|Letter + digit|$x_{2}$|
|`x_12`|`x_{12}`|Letter + two digits|$x_{12}$|
|`xnn`|`x_{n}`|x sub n|$x_{n}$|
|`xii`|`x_{i}`|x sub i|$x_{i}$|
|`xjj`|`x_{j}`|x sub j|$x_{j}$|
|`xp1`|`x_{n+1}`|x sub n+1|$x_{n+1}$|
|`ynn`|`y_{n}`|y sub n|$y_{n}$|
|`yii`|`y_{i}`|y sub i|$y_{i}$|
|`yjj`|`y_{j}`|y sub j|$y_{j}$|

## Font Styles

|Trigger|Replacement|Description|Example|
|---|---|---|---|
|`bf`|`\mathbf{$0}`|Bold|`bf x` → $\mathbf{x}$|
|`rm`|`\mathrm{$0}$1`|Roman|`rm Re` → $\mathrm{Re}$|
|`Re`|`\mathrm{Re}`|Real part|$\mathrm{Re}$|
|`Im`|`\mathrm{Im}`|Imaginary part|$\mathrm{Im}$|

## Accents and Decorations

|Trigger|Replacement|Description|Example|
|---|---|---|---|
|`xhat`|`\hat{x}`|Hat (letter)|$\hat{x}$|
|`hat`|`\hat{$0}$1`|Hat (general)|$\hat{a}$|
|`xbar`|`\bar{x}`|Bar (letter)|$\bar{x}$|
|`bar`|`\bar{$0}$1`|Bar (general)|$\bar{a}$|
|`xdot`|`\dot{x}`|Dot (letter)|$\dot{x}$|
|`dot`|`\dot{$0}$1`|Dot (general)|$\dot{a}$|
|`xddot`|`\ddot{x}`|Double dot (letter)|$\ddot{x}$|
|`ddot`|`\ddot{$0}$1`|Double dot (general)|$\ddot{a}$|
|`xtilde`|`\tilde{x}`|Tilde (letter)|$\tilde{x}$|
|`tilde`|`\tilde{$0}$1`|Tilde (general)|$\tilde{a}$|
|`xund`|`\underline{x}`|Underline (letter)|$\underline{x}$|
|`und`|`\underline{$0}$1`|Underline (general)|$\underline{a}$|
|`xvec`|`\vec{x}`|Vector (letter)|$\vec{x}$|
|`vec`|`\vec{$0}$1`|Vector (general)|$\vec{a}$|
|`x,.` or `x.,`|`\mathbf{x}`|Bold shortcut|$\mathbf{x}$|

## Symbols and Infinity

|Trigger|Replacement|Description|Example|
|---|---|---|---|
|`ooo`|`\infty`|Infinity|$\infty$|
|`...`|`\dots`|Ellipsis|$\dots$|
|`cdot`|`\cdot`|Center dot|$\cdot$|
|`xx`|`\times`|Times|$\times$|
|`**`|`\cdot`|Dot product|$\cdot$|

## Sums, Products, Limits

|Trigger|Replacement|Description|Example|
|---|---|---|---|
|`sum`|`\sum`|Summation|$\sum$|
|`\sum` (in math)|`\sum_{i=1}^{N}`|Sum with limits|$\sum_{i=1}^{N} x_i$|
|`prod`|`\prod`|Product|$\prod$|
|`\prod` (in math)|`\prod_{i=1}^{N}`|Product with limits|$\prod_{i=1}^{N} x_i$|
|`lim`|`\lim_{ n \to \infty }`|Limit|$\lim_{n \to \infty}$|

## Relations and Logic

|Trigger|Replacement|Description|Example|
|---|---|---|---|
|`in`|`\in`|Element of|$\in$|
|`inn`|`\in`|Element of|$\in$|
|`notin`|`\not\in`|Not element of|$\not\in$|
|`sub=`|`\subseteq`|Subset or equal|$\subseteq$|
|`sup=`|`\supseteq`|Superset or equal|$\supseteq$|
|`eset`|`\emptyset`|Empty set|$\emptyset$|
|`set`|`\{ $0 \}$1`|Set brackets|${a, b}$|
|`+-`|`\pm`|Plus-minus|$\pm$|
|`-+`|`\mp`|Minus-plus|$\mp$|
|`!=`|`\neq`|Not equal|$\neq$|
|`>=`|`\geq`|Greater or equal|$\geq$|
|`<=`|`\leq`|Less or equal|$\leq$|
|`>>`|`\gg`|Much greater|$\gg$|
|`<<`|`\ll`|Much less|$\ll$|
|`simm`|`\sim`|Similar|$\sim$|
|`sim=`|`\simeq`|Similar or equal|$\simeq$|
|`===`|`\equiv`|Equivalent|$\equiv$|
|`prop`|`\propto`|Proportional|$\propto$|
|`para`|`\parallel`|Parallel|$\parallel$|

## Arrows

|Trigger|Replacement|Description|Example|
|---|---|---|---|
|`->`|`\to`|Right arrow|$\to$|
|`<->`|`\leftrightarrow`|Left-right arrow|$\leftrightarrow$|
|`!>`|`\mapsto`|Maps to|$\mapsto$|
|`=>`|`\implies`|Implies|$\implies$|
|`=<`|`\impliedby`|Implied by|$\impliedby$|

## Set Operations

|Trigger|Replacement|Description|Example|
|---|---|---|---|
|`and`|`\cap`|Intersection|$\cap$|
|`orr`|`\cup`|Union|$\cup$|
|`\\\`|`\setminus`|Set difference|$\setminus$|

## Common Sets

|Trigger|Replacement|Description|Example|
|---|---|---|---|
|`NN`|`\mathbb{N}`|Natural numbers|$\mathbb{N}$|
|`ZZ`|`\mathbb{Z}`|Integers|$\mathbb{Z}$|
|`RR`|`\mathbb{R}`|Real numbers|$\mathbb{R}$|
|`CC`|`\mathbb{C}`|Complex numbers|$\mathbb{C}$|
|`LL`|`\mathcal{L}`|Lagrangian|$\mathcal{L}$|
|`HH`|`\mathcal{H}`|Hilbert space|$\mathcal{H}$|

## Derivatives and Integrals

|Trigger|Replacement|Description|Example|
|---|---|---|---|
|`par`|`\frac{ \partial y }{ \partial x }`|Partial derivative|$\frac{\partial y}{\partial x}$|
|`paxy`|`\frac{ \partial x }{ \partial y }`|Partial (auto)|$\frac{\partial x}{\partial y}$|
|`ddt`|`\frac{d}{dt}`|Time derivative|$\frac{d}{dt}$|
|`int`|`\int`|Integral|$\int$|
|`\int` (in math)|`\int f \, dx`|Integral with differential|$\int f , dx$|
|`dint`|`\int_{0}^{1} f \, dx`|Definite integral|$\int_{0}^{1} f , dx$|
|`oint`|`\oint`|Contour integral|$\oint$|
|`iint`|`\iint`|Double integral|$\iint$|
|`iiint`|`\iiint`|Triple integral|$\iiint$|
|`oinf`|`\int_{0}^{\infty} f \, dx`|0 to infinity|$\int_{0}^{\infty} f , dx$|
|`infi`|`\int_{-\infty}^{\infty} f \, dx`|-inf to inf|$\int_{-\infty}^{\infty} f , dx$|

## Trigonometry

|Trigger|Replacement|Description|Example|
|---|---|---|---|
|`sin`|`\sin`|Sine|$\sin x$|
|`cos`|`\cos`|Cosine|$\cos x$|
|`tan`|`\tan`|Tangent|$\tan x$|
|`arcsin`|`\arcsin`|Arcsine|$\arcsin x$|
|`arccos`|`\arccos`|Arccosine|$\arccos x$|
|`arctan`|`\arctan`|Arctangent|$\arctan x$|
|`csc`|`\csc`|Cosecant|$\csc x$|
|`sec`|`\sec`|Secant|$\sec x$|
|`cot`|`\cot`|Cotangent|$\cot x$|
|`sinh`|`\sinh`|Hyperbolic sine|$\sinh x$|
|`cosh`|`\cosh`|Hyperbolic cosine|$\cosh x$|
|`tanh`|`\tanh`|Hyperbolic tangent|$\tanh x$|
|`coth`|`\coth`|Hyperbolic cotangent|$\coth x$|

## Functions

|Trigger|Replacement|Description|Example|
|---|---|---|---|
|`exp`|`\exp`|Exponential|$\exp(x)$|
|`log`|`\log`|Logarithm|$\log x$|
|`ln`|`\ln`|Natural log|$\ln x$|
|`det`|`\det`|Determinant|$\det(A)$|
|`trace`|`\mathrm{Tr}`|Trace|$\mathrm{Tr}(A)$|

## Visual Operations (for selected text)
Have to select first and then press.

|Trigger|Replacement|Description|Example|
|---|---|---|---|
|`U`|`\underbrace{ ... }_{ }`|Underbrace|$\underbrace{x+y}_{\text{sum}}$|
|`O`|`\overbrace{ ... }^{ }`|Overbrace|$\overbrace{x+y}^{\text{sum}}$|
|`B`|`\underset{ }{ ... }`|Underset|$\underset{i}{\arg\max}$|
|`C`|`\cancel{ ... }`|Cancel|$\cancel{x}$|
|`K`|`\cancelto{ }{ ... }`|Cancel to|$\cancelto{0}{x}$|
|`S`|`\sqrt{ ... }`|Square root|$\sqrt{x}$|

## Matrix Environments

| Trigger  | Replacement                       | Description         | Example                                     |
| -------- | --------------------------------- | ------------------- | ------------------------------------------- |
| `pmat`   | `\begin{pmatrix}...\end{pmatrix}` | Parenthesis matrix  | $\begin{pmatrix}a & b\\c & d\end{pmatrix}$  |
| `bmat`   | `\begin{bmatrix}...\end{bmatrix}` | Bracket matrix      | $\begin{bmatrix}a & b\\c & d\end{bmatrix}$  |
| `Bmat`   | `\begin{Bmatrix}...\end{Bmatrix}` | Brace matrix        | $\begin{Bmatrix}a & b\\c & d\end{Bmatrix}$  |
| `vmat`   | `\begin{vmatrix}...\end{vmatrix}` | Vertical bar matrix | $\begin{vmatrix}a & b\\c & d\end{vmatrix}$  |
| `Vmat`   | `\begin{Vmatrix}...\end{Vmatrix}` | Double bar matrix   | $\begin{Vmatrix}a & b\\c & d\end{Vmatrix}$  |
| `matrix` | `\begin{matrix}...\end{matrix}`   | Plain matrix        | $\begin{matrix}a & b\\c & d\end{matrix}$    |
| `cases`  | `\begin{cases}...\end{cases}`     | Cases environment   | $\begin{cases}x & x>0\\-x & x<0\end{cases}$ |
| `align`  | `\begin{align}...\end{align}`     | Align environment   | Multi-line equations                        |
| `array`  | `\begin{array}...\end{array}`     | Array environment   | Custom arrays                               |

## Brackets and Delimiters

| Trigger | Replacement           | Description         | Example                        |
| ------- | --------------------- | ------------------- | ------------------------------ |
| `(`     | `($0)$1`              | Parentheses         | $(x)$                          |
| `[`     | `[$0]$1`              | Square brackets     | $[x]$                          |
| `{`     | `{$0}$1`              | Curly braces        | (needs escaping)               |
| `lr(`   | `\left( $0 \right)`   | Auto-sized parens   | $\left(\frac{a}{b}\right)$     |
| `lr[`   | `\left[ $0 \right]`   | Auto-sized brackets | $\left[\frac{a}{b}\right]$     |
| `lr{`   | `\left\{ $0 \right\}` | Auto-sized braces   | $\left\{ \frac{a}{b} \right\}$ |
| `lra`   | `\left< $0 \right>`   | Auto-sized angles   | $\left<\frac{a}{b}\right>$     |
| `avg`   | `\langle $0 \rangle`  | Angle brackets      | $\langle x \rangle$            |
| `norm`  | `\lvert $0 \rvert`    | Norm                | $\lvert x \rvert$              |
| `Norm`  | `\lVert $0 \rVert`    | Double norm         | $\lVert x \rVert$              |
| `ceil`  | `\lceil $0 \rceil`    | Ceiling             | $\lceil x \rceil$              |
| `floor` | `\lfloor $0 \rfloor`  | Floor               | $\lfloor x \rfloor$            |
| `mod`   | `\|$0\|`              | Absolute value      | $                              |

## Physics

|Trigger|Replacement|Description|Example|
|---|---|---|---|
|`nabl` / `del`|`\nabla`|Nabla/Del|$\nabla$|
|`kbt`|`k_{B}T`|Boltzmann constant × T|$k_{B}T$|
|`msun`|`M_{\odot}`|Solar mass|$M_{\odot}$|

## Quantum Mechanics

|Trigger|Replacement|Description|Example|
|---|---|---|---|
|`dag`|`^{\dagger}`|Dagger (Hermitian)|$A^{\dagger}$|
|`o+`|`\oplus`|Direct sum|$\oplus$|
|`ox`|`\otimes`|Tensor product|$\otimes$|
|`bra`|`\bra{$0}`|Bra|$\langle\psi|
|`ket`|`\ket{$0}`|Ket|$|
|`brk`|`\braket{ $0 \| $1 }`|Braket|$\langle\phi|
|`outer`|`\ket{\psi} \bra{\psi}`|Outer product|$|

## Chemistry

|Trigger|Replacement|Description|Example|
|---|---|---|---|
|`pu`|`\pu{ $0 }`|Physical units|$\pu{5 mol}$|
|`cee`|`\ce{ $0 }`|Chemical equation|$\ce{H2O}$|
|`he4`|`{}^{4}_{2}He`|Helium-4|${}^{4}_{2}\text{He}$|
|`he3`|`{}^{3}_{2}He`|Helium-3|${}^{3}_{2}\text{He}$|
|`iso`|`{}^{4}_{2}He`|Isotope notation|${}^{A}_{Z}X$|

## Special Functions

| Trigger | Replacement               | Description        | Result                                                                          |
| ------- | ------------------------- | ------------------ | ------------------------------------------------------------------------------- |
| `tayl`  | Taylor expansion template | Full Taylor series | $f(x+h) = f(x) + f'(x)h + \frac{f''(x)h^2}{2!} + \dots$                         |
| `iden3` | 3×3 identity matrix       | Auto-generates     | $\begin{pmatrix}<br>1 & 0 & 0 \\<br>0 & 1 & 0 \\<br>0 & 0 & 1<br>\end{pmatrix}$ |

This cheatsheet covers all your shortcuts! The most commonly used ones are probably the Greek letters, basic operations (fractions, powers, subscripts), and the matrix environments.