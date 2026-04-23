/// RPN expression tree node.
#[derive(Clone, Debug, PartialEq)]
pub enum Node {
    Const(f64),        // fixed constant
    Param(usize, f64), // tunable parameter: (id, initial_value)
    Var(usize),        // input variable index
    Op(Op),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Op {
    // Unary
    Neg, Inv, Exp, Ln, Sqrt, Sin, Cos, Abs,
    // Binary
    Eml,  // exp(a) - ln(b)  — the Sheffer primitive
    Add, Sub, Mul, Div,
}

impl Op {
    pub fn arity(&self) -> usize {
        match self {
            Op::Neg | Op::Inv | Op::Exp | Op::Ln | Op::Sqrt | Op::Sin | Op::Cos | Op::Abs => 1,
            Op::Eml | Op::Add | Op::Sub | Op::Mul | Op::Div => 2,
        }
    }
}

/// Flat RPN expression: evaluated left-to-right with a stack.
#[derive(Clone, Debug)]
pub struct Expr {
    pub nodes: Vec<Node>,
    pub n_vars: usize,
}

impl Expr {
    pub fn new_const(v: f64) -> Self {
        Expr { nodes: vec![Node::Const(v)], n_vars: 0 }
    }

    pub fn new_var(idx: usize) -> Self {
        Expr { nodes: vec![Node::Var(idx)], n_vars: idx + 1 }
    }

    pub fn complexity(&self) -> usize {
        self.nodes.len()
    }

    /// Evaluate at a single point. vars must have length >= n_vars.
    pub fn eval(&self, vars: &[f64], params: &[f64]) -> Option<f64> {
        let mut stack: Vec<f64> = Vec::with_capacity(self.nodes.len());
        for node in &self.nodes {
            match node {
                Node::Const(v) => stack.push(*v),
                Node::Param(id, init) => {
                    let v = params.get(*id).copied().unwrap_or(*init);
                    stack.push(v);
                }
                Node::Var(i) => {
                    stack.push(*vars.get(*i)?);
                }
                Node::Op(op) => {
                    if stack.len() < op.arity() { return None; }
                    let result = match op {
                        Op::Neg  => { let a = stack.pop()?; -a }
                        Op::Inv  => { let a = stack.pop()?; if a.abs() < 1e-300 { return None; } 1.0 / a }
                        Op::Exp  => { let a = stack.pop()?; a.exp() }
                        Op::Ln   => { let a = stack.pop()?; if a <= 0.0 { return None; } a.ln() }
                        Op::Sqrt => { let a = stack.pop()?; if a < 0.0 { return None; } a.sqrt() }
                        Op::Sin  => { let a = stack.pop()?; a.sin() }
                        Op::Cos  => { let a = stack.pop()?; a.cos() }
                        Op::Abs  => { let a = stack.pop()?; a.abs() }
                        Op::Eml  => {
                            let b = stack.pop()?;
                            let a = stack.pop()?;
                            let xv = if a > 709.78 { a.ln() } else { a };
                            let y_safe = if b <= 0.0 { b.abs().max(1e-300) } else { b };
                            xv.exp() - y_safe.ln()
                        }
                        Op::Add  => { let b = stack.pop()?; let a = stack.pop()?; a + b }
                        Op::Sub  => { let b = stack.pop()?; let a = stack.pop()?; a - b }
                        Op::Mul  => { let b = stack.pop()?; let a = stack.pop()?; a * b }
                        Op::Div  => { let b = stack.pop()?; let a = stack.pop()?; if b.abs() < 1e-300 { return None; } a / b }
                    };
                    if !result.is_finite() { return None; }
                    stack.push(result);
                }
            }
        }
        if stack.len() == 1 { Some(stack[0]) } else { None }
    }

    /// Evaluate at all data points. Returns None if any eval fails.
    pub fn eval_batch(&self, data: &[Vec<f64>], params: &[f64]) -> Option<Vec<f64>> {
        let n = data[0].len();
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let vars: Vec<f64> = data.iter().map(|col| col[i]).collect();
            out.push(self.eval(&vars, params)?);
        }
        Some(out)
    }

    /// Human-readable infix representation.
    pub fn to_string(&self) -> String {
        let mut stack: Vec<String> = Vec::new();
        for node in &self.nodes {
            match node {
                Node::Const(v) => {
                    let s = if *v == std::f64::consts::PI { "π".to_string() }
                            else if *v == std::f64::consts::E { "e".to_string() }
                            else { format!("{:.4}", v) };
                    stack.push(s);
                }
                Node::Param(id, v) => stack.push(format!("c{}({:.4})", id, v)),
                Node::Var(i) => {
                    let name = (b'x' + *i as u8) as char;
                    stack.push(name.to_string());
                }
                Node::Op(op) => {
                    match op {
                        Op::Neg  => { let a = stack.pop().unwrap_or_default(); stack.push(format!("(-{})", a)); }
                        Op::Inv  => { let a = stack.pop().unwrap_or_default(); stack.push(format!("(1/{})", a)); }
                        Op::Exp  => { let a = stack.pop().unwrap_or_default(); stack.push(format!("exp({})", a)); }
                        Op::Ln   => { let a = stack.pop().unwrap_or_default(); stack.push(format!("ln({})", a)); }
                        Op::Sqrt => { let a = stack.pop().unwrap_or_default(); stack.push(format!("sqrt({})", a)); }
                        Op::Sin  => { let a = stack.pop().unwrap_or_default(); stack.push(format!("sin({})", a)); }
                        Op::Cos  => { let a = stack.pop().unwrap_or_default(); stack.push(format!("cos({})", a)); }
                        Op::Abs  => { let a = stack.pop().unwrap_or_default(); stack.push(format!("abs({})", a)); }
                        Op::Eml  => { let b = stack.pop().unwrap_or_default(); let a = stack.pop().unwrap_or_default(); stack.push(format!("eml({}, {})", a, b)); }
                        Op::Add  => { let b = stack.pop().unwrap_or_default(); let a = stack.pop().unwrap_or_default(); stack.push(format!("({} + {})", a, b)); }
                        Op::Sub  => { let b = stack.pop().unwrap_or_default(); let a = stack.pop().unwrap_or_default(); stack.push(format!("({} - {})", a, b)); }
                        Op::Mul  => { let b = stack.pop().unwrap_or_default(); let a = stack.pop().unwrap_or_default(); stack.push(format!("({} * {})", a, b)); }
                        Op::Div  => { let b = stack.pop().unwrap_or_default(); let a = stack.pop().unwrap_or_default(); stack.push(format!("({} / {})", a, b)); }
                    }
                }
            }
        }
        stack.pop().unwrap_or_else(|| "?".to_string())
    }

    /// Collect mutable parameter initial values.
    pub fn param_count(&self) -> usize {
        self.nodes.iter().filter_map(|n| if let Node::Param(id, _) = n { Some(*id + 1) } else { None }).max().unwrap_or(0)
    }
}
