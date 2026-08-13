import graphviz
from IPython.display import display

dot_data = """
digraph BERTDetailedArchitecture {
    // Configurações do Grafo para compressão e alinhamento
    rankdir=LR;
    nodesep=0.3; // Aproxima os nós verticalmente
    ranksep=0.6; // Comprime a distância horizontal entre as camadas
    splines=polyline; // Arestas retas minimizam curvas que ocupam espaço
    fontname="Helvetica";

    node [shape=plain, fontname="Helvetica", fontsize=11];
    edge [color="#495057", penwidth=1.2, arrowsize=0.8];

    // Cluster de Entrada (Alinhado Verticalmente)
    subgraph cluster_input_features {
        label="Input Features (Max Seq Length: 128)";
        style="dashed"; color="#adb5bd"; fontname="Helvetica-Bold"; fontsize=12;
        rank=same;

        TokenIds [label=<
        <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="5">
            <TR><TD BGCOLOR="#e9ecef"><B>token_ids</B></TD></TR>
            <TR><TD>InputLayer</TD></TR>
            <TR><TD>Shape: (None, 128) | dtype: int32</TD></TR>
        </TABLE>>];

        SegmentIds [label=<
        <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="5">
            <TR><TD BGCOLOR="#e9ecef"><B>segment_ids</B></TD></TR>
            <TR><TD>InputLayer</TD></TR>
            <TR><TD>Shape: (None, 128) | dtype: int32</TD></TR>
        </TABLE>>];

        PaddingMask [label=<
        <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="5">
            <TR><TD BGCOLOR="#e9ecef"><B>padding_mask</B></TD></TR>
            <TR><TD>InputLayer</TD></TR>
            <TR><TD>Shape: (None, 128) | dtype: bool</TD></TR>
        </TABLE>>];
    }

    // Camadas de Processamento
    BertBackbone [label=<
    <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="5">
        <TR><TD BGCOLOR="#d0ebff"><B>bert_backbone</B></TD></TR>
        <TR><TD>BertBackbone (bert_base_en_uncased)</TD></TR>
        <TR><TD>Params: 109,482,240</TD></TR>
        <TR><TD>Output Shape: (None, 768) [pooled_output]</TD></TR>
    </TABLE>>];

    DropoutLayer [label=<
    <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="5">
        <TR><TD BGCOLOR="#fff3cd"><B>dropout_25</B></TD></TR>
        <TR><TD>Dropout</TD></TR>
        <TR><TD>Rate: 0.1</TD></TR>
        <TR><TD>Output Shape: (None, 768)</TD></TR>
    </TABLE>>];

    Classifier [label=<
    <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="5">
        <TR><TD BGCOLOR="#d4edda"><B>classifier</B></TD></TR>
        <TR><TD>Dense (from_logits=True)</TD></TR>
        <TR><TD>Params: 1,538</TD></TR>
        <TR><TD>Output Shape: (None, 2)</TD></TR>
    </TABLE>>];

    // Conexões
    TokenIds -> BertBackbone;
    SegmentIds -> BertBackbone;
    PaddingMask -> BertBackbone;

    BertBackbone -> DropoutLayer;
    DropoutLayer -> Classifier;
}
"""

graph = graphviz.Source(dot_data)

graph.render(
    filename="bert_architecture_imdb",
    format="svg",
    cleanup=True
)