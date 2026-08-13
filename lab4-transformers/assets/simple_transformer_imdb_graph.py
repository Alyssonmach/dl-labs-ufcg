import graphviz

dot_data = """
digraph Transformer_IMDB_Review {
    rankdir=LR;
    splines=ortho;
    nodesep=0.15;
    ranksep=0.25;

    node [
        shape=box,
        style="filled,rounded",
        fontname="Helvetica",
        fontsize=10,
        margin="0.05,0.05",
        height=0.3
    ];

    edge [
        fontname="Helvetica",
        fontsize=8,
        color="#555555"
    ];

    // Pré-Processamento
    subgraph cluster_preprocessing {
        style=dashed;
        color="#cccccc";
        label="Pré-Processamento";
        fontname="Helvetica-Bold";
        fontsize=10;

        Input [
            label="Entrada:\\nX₁ ... Xₙ",
            fillcolor="#d5e8d4",
            color="#82b366"
        ];

        TokenEmb [
            label="Embedding\\nde Tokens",
            fillcolor="#dae8fc",
            color="#6c8ebf"
        ];

        PosEmb [
            label="Embedding\\nPosicional",
            fillcolor="#fff2cc",
            color="#d6b656"
        ];

        AddEmb [
            label="⊕",
            shape=circle,
            fillcolor="#f5f5f5",
            color="#666666",
            width=0.25,
            fixedsize=true
        ];

        Input -> TokenEmb;
        TokenEmb -> AddEmb;
        PosEmb -> AddEmb;
    }

    // Encoder
    subgraph cluster_encoder {
        style="dashed,rounded";
        color="#b85450";
        label="Encoder";
        fontname="Helvetica-Bold";
        fontsize=10;

        MHA [
            label="Auto-\\natenção",
            fillcolor="#ffe6cc",
            color="#d79b00"
        ];

        AddNorm1 [
            label="Soma &\\nNorm",
            fillcolor="#e1d5e7",
            color="#9673a6"
        ];

        FFN [
            label="Feed-\\nForward",
            fillcolor="#dae8fc",
            color="#6c8ebf"
        ];

        AddNorm2 [
            label="Soma &\\nNorm",
            fillcolor="#e1d5e7",
            color="#9673a6"
        ];

        MHA -> AddNorm1 [style=bold];
        AddNorm1 -> FFN [style=bold];
        FFN -> AddNorm2 [style=bold];
    }

    // Saída
    subgraph cluster_classification {
        style=dashed;
        color="#cccccc";
        label="Saída";
        fontname="Helvetica-Bold";
        fontsize=10;

        Pool [
            label="Avg\\nPool",
            fillcolor="#d5e8d4",
            color="#82b366"
        ];

        Output [
            label="Linear\\n(+)/(-)",
            fillcolor="#f8cecc",
            color="#b85450",
            shape=ellipse
        ];

        Pool -> Output [style=bold];
    }

    // Fluxo principal
    AddEmb -> MHA [
        style=bold,
        label=" (batch, 200, 32)"
    ];

    AddNorm2 -> Pool [style=bold];

    // Conexões residuais
    AddEmb -> AddNorm1 [
        style=dashed,
        color="#b85450",
        label=" Atalho",
        constraint=false
    ];

    AddNorm1 -> AddNorm2 [
        style=dashed,
        color="#b85450",
        label=" Atalho",
        constraint=false
    ];
}
"""

graph = graphviz.Source(dot_data)


graph.render(
    filename="transformer_imdb_architecture",
    format="svg",
    cleanup=True
)