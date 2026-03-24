#!/usr/bin/env python3
"""Generate a PDF of the OLS-in-Deep-Learning conversation."""

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable
)

OUTPUT = "/Users/giladsenderovich/olsvered/OLS_in_Deep_Learning_Conversation.pdf"

def build_styles():
    ss = getSampleStyleSheet()

    ss.add(ParagraphStyle(
        "DocTitle", parent=ss["Title"], fontSize=22, spaceAfter=6,
        textColor=HexColor("#1a1a2e"),
    ))
    ss.add(ParagraphStyle(
        "Subtitle", parent=ss["Normal"], fontSize=11,
        textColor=HexColor("#666666"), alignment=TA_CENTER, spaceAfter=20,
    ))
    ss.add(ParagraphStyle(
        "H1", parent=ss["Heading1"], fontSize=16, spaceAfter=8,
        spaceBefore=18, textColor=HexColor("#1a1a2e"),
    ))
    ss.add(ParagraphStyle(
        "H2", parent=ss["Heading2"], fontSize=13, spaceAfter=6,
        spaceBefore=14, textColor=HexColor("#2d3436"),
    ))
    ss.add(ParagraphStyle(
        "H3", parent=ss["Heading3"], fontSize=11, spaceAfter=4,
        spaceBefore=10, textColor=HexColor("#444444"),
    ))
    ss.add(ParagraphStyle(
        "Body", parent=ss["Normal"], fontSize=10, leading=14,
        spaceAfter=6,
    ))
    ss.add(ParagraphStyle(
        "BulletCustom", parent=ss["Normal"], fontSize=10, leading=14,
        leftIndent=20, spaceAfter=3, bulletIndent=10,
    ))
    ss.add(ParagraphStyle(
        "CodeBlock", parent=ss["Normal"], fontSize=8.5, leading=11,
        fontName="Courier", backColor=HexColor("#f5f5f5"),
        leftIndent=12, rightIndent=12, spaceAfter=8, spaceBefore=4,
    ))
    ss.add(ParagraphStyle(
        "Question", parent=ss["Normal"], fontSize=11, leading=15,
        textColor=HexColor("#0066cc"), fontName="Helvetica-Bold",
        spaceBefore=12, spaceAfter=8, leftIndent=10,
        borderColor=HexColor("#0066cc"), borderWidth=0,
        backColor=HexColor("#e8f0fe"), borderPadding=8,
    ))
    ss.add(ParagraphStyle(
        "TableCell", parent=ss["Normal"], fontSize=8.5, leading=11,
    ))
    ss.add(ParagraphStyle(
        "TableHeader", parent=ss["Normal"], fontSize=8.5, leading=11,
        fontName="Helvetica-Bold", textColor=HexColor("#ffffff"),
    ))
    return ss


def make_table(headers, rows, col_widths=None):
    """Build a styled Table from header list and row list."""
    s = getSampleStyleSheet()
    hdr_style = ParagraphStyle("_th", parent=s["Normal"], fontSize=8.5,
                                fontName="Helvetica-Bold",
                                textColor=HexColor("#ffffff"), leading=11)
    cell_style = ParagraphStyle("_td", parent=s["Normal"], fontSize=8.5, leading=11)

    data = [[Paragraph(h, hdr_style) for h in headers]]
    for row in rows:
        data.append([Paragraph(str(c), cell_style) for c in row])

    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#2d3436")),
        ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#ffffff")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
        ("TOPPADDING", (0, 0), (-1, 0), 8),
        ("BACKGROUND", (0, 1), (-1, -1), HexColor("#fafafa")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#ffffff"), HexColor("#f0f0f0")]),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cccccc")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    return t


def hr():
    return HRFlowable(width="100%", thickness=0.5, color=HexColor("#cccccc"),
                       spaceBefore=8, spaceAfter=8)


def build_pdf():
    doc = SimpleDocTemplate(
        OUTPUT, pagesize=letter,
        leftMargin=0.75*inch, rightMargin=0.75*inch,
        topMargin=0.75*inch, bottomMargin=0.75*inch,
        title="OLS Algorithms in Deep Learning Training",
        author="Gilad Senderovich & Claude",
    )
    s = build_styles()
    story = []

    # ── Title page ──
    story.append(Spacer(1, 60))
    story.append(Paragraph("OLS Algorithms in Deep Learning Training", s["DocTitle"]))
    story.append(Paragraph(
        "A conversation exploring how the olsvered closed-form OLS algorithms<br/>"
        "can be applied to deep learning architectures", s["Subtitle"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph(
        "Based on: <i>Solving The Ordinary Least Squares in Closed Form, Without "
        "Inversion or Normalization</i><br/>Vered Senderovich Madar &amp; Sandra L. Batista "
        "-- arXiv:2301.01854", s["Subtitle"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph("March 2026", s["Subtitle"]))
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════
    # Section 1: Can OLS Be Used in Deep Learning?
    # ══════════════════════════════════════════════════════════════════
    story.append(Paragraph("1. Can OLS Solving Algorithms Be Used in Training Deep Learning?", s["H1"]))
    story.append(Paragraph(
        "<b>Short answer: Yes</b> -- for specific, well-defined sub-problems within DL architectures.", s["Body"]))

    story.append(Paragraph("1.1 Where OLS Appears Naturally in Deep Learning", s["H2"]))
    story.append(make_table(
        ["DL Scenario", "OLS Role"],
        [
            ["Linear output heads", "Solve for last-layer weights with frozen backbone"],
            ["Linear probing", "One-shot closed-form solve for transfer learning evaluation"],
            ["Extreme Learning Machines", "Random hidden layers; OLS for output weights"],
            ["Echo State Networks", "Readout weight training = pure ridge regression"],
            ["Meta-learning (few-shot)", "Tiny per-task OLS problem (e.g. 5-shot x 5-way = 25 samples)"],
            ["NTK regime", "Infinite-width DL reduces to kernel ridge regression"],
        ],
        col_widths=[2.2*inch, 4.8*inch],
    ))
    story.append(Spacer(1, 8))

    story.append(Paragraph("1.2 How the 3 olsvered Algorithms Map", s["H2"]))
    story.append(make_table(
        ["Algorithm", "Best DL Use Case"],
        [
            ["Alg 1 -- Modified Cholesky",
             "Last-layer optimization, linear probing, ELM readouts, meta-learning base learner"],
            ["Alg 2 -- SGSO (unnorm. GS)",
             "Weight orthogonalization, feature decorrelation, continual learning, orthogonal init"],
            ["Alg 3 -- Weighted Gen. Inverse",
             "Importance-weighted training, curriculum learning, domain adaptation, heteroscedastic regression"],
        ],
        col_widths=[2.2*inch, 4.8*inch],
    ))
    story.append(Spacer(1, 8))

    story.append(Paragraph("1.3 Most Promising Research Direction", s["H2"]))
    story.append(Paragraph(
        "<b>Alternating Closed-Form / Gradient Optimization</b> (Galashov et al., 2025): "
        "every K gradient steps, recompute last-layer weights via Algorithm 1 in closed form. "
        "Accumulate X<super>T</super>X and X<super>T</super>y as running statistics across mini-batches. "
        "Extend with Algorithm 3 for sample-weighted variants (curriculum, class imbalance).", s["Body"]))

    story.append(Paragraph("1.4 Key Limitations", s["H2"]))
    story.append(make_table(
        ["Limitation", "Notes"],
        [
            ["Non-linearity", "OLS only applies to linear sub-problems -- fundamental boundary"],
            ["Scale", "Gram matrix (p x p): practical up to ~p=8192 on CPU"],
            ["Mini-batch staleness", "Features change as backbone updates; accumulated stats go stale"],
            ["No L1 regularization", "Lasso has no closed form; ridge (X<sup>T</sup>X + lambda I) is trivial"],
            ["Condition number", "cond(X<sup>T</sup>X) = cond(X)<sup>2</sup> -- use float64"],
        ],
        col_widths=[1.8*inch, 5.2*inch],
    ))
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════
    # Section 2: Gram Matrix Scalability
    # ══════════════════════════════════════════════════════════════════
    story.append(Paragraph("2. Gram Matrix Scalability Explained", s["H1"]))

    story.append(Paragraph("2.1 What Is the Gram Matrix?", s["H2"]))
    story.append(Paragraph(
        "In OLS you solve beta = (X<super>T</super>X)<super>-1</super>X<super>T</super>y. "
        "The term X<super>T</super>X is the <b>Gram matrix</b>. "
        "If X has shape (n, p) then X<super>T</super>X has shape <b>(p x p)</b>, "
        "regardless of the number of samples.", s["Body"]))

    story.append(Paragraph("2.2 Why p Matters More Than n", s["H2"]))
    story.append(Paragraph(
        "In deep learning, p is the feature/hidden dimension of the layer being solved:", s["Body"]))
    story.append(make_table(
        ["Architecture", "Typical p", "Gram Matrix", "Memory (float64)"],
        [
            ["Small CNN head", "512", "512 x 512", "2 MB"],
            ["ResNet-50", "2,048", "2,048 x 2,048", "32 MB"],
            ["ViT-Large", "1,024", "1,024 x 1,024", "8 MB"],
            ["GPT-2 medium", "4,096", "4,096 x 4,096", "128 MB"],
            ["LLaMA-7B", "4,096", "4,096 x 4,096", "128 MB"],
            ["LLaMA-65B", "8,192", "8,192 x 8,192", "512 MB"],
            ["GPT-4 class", "~12,288+", "12,288 x 12,288", "1.1 GB"],
            ["Large MoE", "65,536", "65,536 x 65,536", "32 GB"],
        ],
        col_widths=[1.6*inch, 1.2*inch, 1.8*inch, 1.5*inch],
    ))
    story.append(Spacer(1, 8))

    story.append(Paragraph("2.3 The O(p<super>3</super>) Bottleneck", s["H2"]))
    story.append(Paragraph(
        "LU decomposition costs O(p<super>3</super>) FLOPs. Double p and compute grows 8x:", s["Body"]))
    story.append(make_table(
        ["p", "FLOPs", "Wall Time (approx)"],
        [
            ["512", "~134 million", "milliseconds"],
            ["2,048", "~8.6 billion", "~1 second"],
            ["8,192", "~550 billion", "~30-60 seconds"],
            ["65,536", "~281 trillion", "hours on CPU"],
        ],
        col_widths=[1.5*inch, 2.5*inch, 2.5*inch],
    ))
    story.append(Spacer(1, 8))

    story.append(Paragraph("2.4 Practical Boundaries", s["H2"]))
    story.append(make_table(
        ["Range", "Assessment"],
        [
            ["p &lt;= 2,048", "Instant -- use freely (covers most vision models)"],
            ["p &lt;= 4,096", "Fast -- use every few gradient steps (covers most LLMs)"],
            ["p &lt;= 8,192", "Feasible -- use sparingly (large LLMs)"],
            ["p &gt; 8,192", "Switch to iterative methods (CG, truncated SVD, sketching)"],
        ],
        col_widths=[1.5*inch, 5.5*inch],
    ))
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════
    # Section 3: Which Algorithms Use the Gram Matrix?
    # ══════════════════════════════════════════════════════════════════
    story.append(Paragraph("3. Gram Matrix Usage by Algorithm", s["H1"]))

    story.append(make_table(
        ["Algorithm", "Uses Gram Matrix?", "What It Computes", "Shape"],
        [
            ["Alg 1 -- Modified Cholesky", "Yes", "G = [X|y]<sup>T</sup>[X|y] (augmented)", "(p+1) x (p+1)"],
            ["Alg 2 -- SGSO", "No", "Column-wise dot products on X directly", "operates on (n, p)"],
            ["Alg 3 -- Weighted Gen. Inverse", "Yes", "X<sup>T</sup>WX (weighted Gram)", "p x p"],
        ],
        col_widths=[1.8*inch, 1.2*inch, 2.5*inch, 1.5*inch],
    ))
    story.append(Spacer(1, 8))

    story.append(Paragraph("3.1 Algorithm 1 (line 75 in algorithms.rs)", s["H2"]))
    story.append(Paragraph("let gram = xy.transpose() * &amp;xy; &nbsp; // (p+1) x (p+1)", s["CodeBlock"]))
    story.append(Paragraph(
        "Forms the augmented Gram matrix [X|y]<super>T</super>[X|y], then LU-decomposes it. "
        "Bounded by O(p<super>3</super>).", s["Body"]))

    story.append(Paragraph("3.2 Algorithm 2 (lines 186-202)", s["H2"]))
    story.append(Paragraph(
        "let num = qj.dot(&amp;qi); &nbsp; // scalar dot product<br/>"
        "let den = qi.dot(&amp;qi); &nbsp; // scalar dot product", s["CodeBlock"]))
    story.append(Paragraph(
        "<b>No Gram matrix at all.</b> Works column-by-column with dot products directly on "
        "the (n, p) matrix. Cost is O(np<super>2</super>) -- scales with the number of samples.", s["Body"]))

    story.append(Paragraph("3.3 Algorithm 3 (line 243)", s["H2"]))
    story.append(Paragraph("let xtwx = &amp;xtw * x; &nbsp; // p x p", s["CodeBlock"]))
    story.append(Paragraph(
        "Forms the weighted Gram matrix X<super>T</super>WX, then solves via LU. "
        "Same O(p<super>3</super>) bound as Algorithm 1.", s["Body"]))

    story.append(Paragraph(
        "<b>Bottom line:</b> The p=8,192 wall applies to <b>Algorithms 1 and 3</b> only. "
        "Algorithm 2 (SGSO) has different scaling -- O(np<super>2</super>).", s["Body"]))
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════
    # Section 4: n and p
    # ══════════════════════════════════════════════════════════════════
    story.append(Paragraph("4. What Do n and p Stand For?", s["H1"]))
    story.append(make_table(
        ["Symbol", "Meaning", "Example"],
        [
            ["n", "Number of observations (rows / samples / data points)", "1,000 patients in a study"],
            ["p", "Number of predictors (columns / features / variables)", "10 measurements per patient"],
        ],
        col_widths=[0.8*inch, 3.2*inch, 3*inch],
    ))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "The design matrix X has shape <b>(n x p)</b>. In deep learning terms: "
        "n = batch/dataset size, p = hidden/feature dimension of a layer.", s["Body"]))
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════
    # Section 5: When Can OLS Replace Gradient Descent?
    # ══════════════════════════════════════════════════════════════════
    story.append(Paragraph("5. When Can Closed-Form OLS Replace Gradient Descent?", s["H1"]))

    story.append(Paragraph("5.1 The One Rule", s["H2"]))
    story.append(Paragraph(
        "<b>If the layer is linear and the loss is squared error, you can replace gradient "
        "descent with a direct solve.</b> Gradient descent <i>iterates</i> toward the answer. "
        "OLS <i>computes</i> it in one shot.", s["Body"]))

    story.append(Paragraph("5.2 Where It Can Replace GD Today", s["H2"]))
    story.append(make_table(
        ["Scenario", "Why OLS Works", "GD Steps Replaced"],
        [
            ["Last linear layer (frozen backbone)", "It is literally y = X beta", "Hundreds/thousands of SGD steps -> 1 solve"],
            ["Linear probing", "Frozen features, linear head", "Entire training loop -> 1 solve"],
            ["ELM / Reservoir readout", "Random fixed layers, only readout trained", "All training -> 1 solve"],
            ["Few-shot meta-learning", "Tiny linear problem per task", "Per-task fine-tuning -> instant solve"],
        ],
        col_widths=[1.8*inch, 2.2*inch, 3*inch],
    ))
    story.append(Spacer(1, 8))

    story.append(Paragraph("5.3 The Practical Payoff", s["H2"]))
    story.append(make_table(
        ["Metric", "SGD on Last Layer", "OLS Solve"],
        [
            ["Steps to converge", "100 -- 10,000", "1"],
            ["Result", "Approximate", "Exact (to machine precision)"],
            ["Hyperparameters", "LR, momentum, schedule", "None"],
            ["When p &lt;= 4096", "Minutes of training", "Milliseconds"],
        ],
        col_widths=[1.8*inch, 2.6*inch, 2.6*inch],
    ))
    story.append(Spacer(1, 8))

    story.append(Paragraph("5.4 Where It Cannot Replace GD", s["H2"]))
    story.append(make_table(
        ["Situation", "Why Not"],
        [
            ["Non-linear layers (ReLU, attention)", "No closed-form solution; loss landscape is non-convex"],
            ["Cross-entropy loss", "OLS requires squared error; classification uses log-loss"],
            ["Regularization beyond L2", "Dropout, L1, batch norm have no closed form"],
            ["n &lt; p (more features than samples)", "Gram matrix is singular; need ridge regression"],
            ["Streaming / online learning", "OLS needs all data at once (mitigated by incremental X<sup>T</sup>X)"],
        ],
        col_widths=[2.5*inch, 4.5*inch],
    ))
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════
    # Section 6: Unique Efficiency Advantages
    # ══════════════════════════════════════════════════════════════════
    story.append(Paragraph(
        "6. Where Do These Algorithms Unlock OLS Due to Better Efficiency?", s["H1"]))

    story.append(Paragraph("6.1 Gram + LU vs Standard SVD", s["H2"]))
    story.append(Paragraph(
        "Standard numpy.linalg.lstsq uses SVD at O(np<super>2</super>) on the full (n, p) matrix. "
        "Algorithms 1 &amp; 3 first compress to the (p, p) Gram matrix then solve via LU "
        "at O(np + p<super>3</super>):", s["Body"]))
    story.append(make_table(
        ["n (samples)", "p (features)", "SVD Cost", "Gram + LU Cost", "Speedup"],
        [
            ["10,000", "64", "41M", "0.9M", "~45x"],
            ["100,000", "128", "1.6B", "15M", "~107x"],
            ["1,000,000", "256", "65B", "273M", "~238x"],
            ["10,000", "2,048", "42B", "29B", "~1.4x"],
        ],
        col_widths=[1.2*inch, 1.1*inch, 1.2*inch, 1.4*inch, 1.1*inch],
    ))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "<b>Sweet spot: n >> p</b> (many samples, moderate feature dimension). "
        "The bigger the ratio n/p, the larger the advantage.", s["Body"]))

    story.append(Paragraph("6.2 Algorithm 1 -- OLS Inside Training Loops", s["H2"]))
    story.append(Paragraph(
        "With incremental Gram accumulation, the entire dataset is compressed into a (p x p) matrix. "
        "Each solve is only O(p<super>3</super>), independent of total n. "
        "If each solve takes 50ms instead of 500ms, you can run it 10x more frequently.", s["Body"]))
    story.append(Paragraph(
        "# Running Gram matrix -- O(batch_size * p) per step<br/>"
        "G += X_batch.T @ X_batch &nbsp;&nbsp; # accumulate (p x p)<br/>"
        "h += X_batch.T @ y_batch &nbsp;&nbsp; # accumulate (p,)<br/><br/>"
        "# Every K steps -- solve is only O(p**3)<br/>"
        "beta = lu_solve(G, h) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; # Algorithm 1 core", s["CodeBlock"]))

    story.append(Paragraph("6.3 Algorithm 2 -- Differentiable Orthogonalization Layer (Unique Unlock)", s["H2"]))
    story.append(Paragraph(
        "Standard Gram-Schmidt has square roots in normalization. Backpropagating through "
        "1/sqrt(x) gives d/dx = -1/(2*x<super>3/2</super>), which <b>explodes when x approaches 0</b>.", s["Body"]))
    story.append(Paragraph(
        "<b>Algorithm 2 (SGSO) has no square roots.</b> Its backward pass only involves divisions "
        "by dot products q<sub>i</sub><super>T</super>q<sub>i</sub>, which are more numerically stable. "
        "This means it can be used as a differentiable layer inside a neural network with stable gradients.", s["Body"]))
    story.append(Paragraph("Use cases this unlocks:", s["Body"]))
    story.append(Paragraph("- Orthogonal weight constraints as a cheap forward-pass operation", s["BulletCustom"]))
    story.append(Paragraph("- Feature decorrelation layers that are differentiable and stable", s["BulletCustom"]))
    story.append(Paragraph("- Continual learning: project gradients onto orthogonal complements during training", s["BulletCustom"]))

    story.append(Paragraph("6.4 Algorithm 3 -- Attention-Weighted Readouts (Practical Unlock)", s["H2"]))
    story.append(Paragraph(
        "When an attention mechanism produces per-sample weights, Algorithm 3 computes the "
        "weighted-optimal linear projection via LU solve <b>without explicit matrix inversion</b>. "
        "This makes weighted OLS cheap and safe enough to run per-attention-head, per-layer.", s["Body"]))

    story.append(Spacer(1, 12))
    story.append(Paragraph("6.5 Summary: Unique Edge per Algorithm", s["H2"]))
    story.append(make_table(
        ["Algorithm", "Unique Efficiency Advantage", "What It Unlocks"],
        [
            ["Alg 1 -- Modified Cholesky",
             "O(np + p<sup>3</sup>) vs O(np<sup>2</sup>); incremental Gram accumulation",
             "OLS in training inner loops, frequent last-layer updates"],
            ["Alg 2 -- SGSO",
             "No sqrt -> stable gradients; non-iterative",
             "Differentiable orthogonalization as a network layer"],
            ["Alg 3 -- Weighted GI",
             "Weighted OLS via LU (no explicit inversion)",
             "Per-head attention-weighted readouts; stable weighted solves"],
        ],
        col_widths=[1.5*inch, 2.5*inch, 3*inch],
    ))
    story.append(Spacer(1, 12))
    story.append(Paragraph(
        "<b>Overarching theme:</b> these algorithms make OLS cheap enough and stable enough "
        "to run <i>inside</i> a forward/backward pass, not just as a one-time offline computation.", s["Body"]))

    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════
    # References
    # ══════════════════════════════════════════════════════════════════
    story.append(Paragraph("References", s["H1"]))
    refs = [
        "Senderovich Madar, V. &amp; Batista, S. (2023). Solving The Ordinary Least Squares in Closed Form, Without Inversion or Normalization. arXiv:2301.01854.",
        "Galashov, A. et al. (2025). Closed-Form Last Layer Optimization. arXiv:2510.04606.",
        "Bertinetto, L. et al. (2019). Meta-Learning with Differentiable Closed-Form Solvers. ICLR 2019.",
        "Kumar, A. et al. (2024). Understanding Linear Probing then Fine-tuning. NeurIPS 2024.",
        "Non-iterative CNN Training via Gram-Schmidt Process. Neural Processing Letters, 2025.",
        "M-estimation ELM Ensembles. Nature Scientific Reports, 2025.",
        "Monga, V. et al. (2019). Algorithm Unrolling: Interpretable, Efficient Deep Learning. arXiv:1912.10557.",
        "Overcoming Catastrophic Interference Using Gram-Schmidt Orthogonalization. PLOS ONE, 2014.",
    ]
    for i, ref in enumerate(refs, 1):
        story.append(Paragraph(f"[{i}] {ref}", s["Body"]))

    # Build
    doc.build(story)
    print(f"PDF created: {OUTPUT}")


if __name__ == "__main__":
    build_pdf()
