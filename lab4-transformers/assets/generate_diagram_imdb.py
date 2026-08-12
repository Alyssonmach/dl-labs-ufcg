import graphviz
from IPython.display import display

def transformer_diagram():
    # gera um diagrama do transformer implementando com o graphviz
    
    dot = graphviz.Digraph(name='Transformer_IMDB_Review', comment='Simple Architecture')
   
    dot.attr(rankdir='LR', splines='ortho', nodesep='0.15', ranksep='0.25')
    
    dot.attr('node', shape='box', style='filled,rounded', fontname='Helvetica', 
             fontsize='10', margin='0.05,0.05', height='0.3')
    dot.attr('edge', fontname='Helvetica', fontsize='8', color='#555555')

    # parte do pré-processamento
    with dot.subgraph(name='cluster_preprocessing') as prep:
        prep.attr(style='dashed', color='#cccccc', label='Pré-Processamento', fontname='Helvetica-Bold', fontsize='10')
        prep.node('Input', 'Entrada:\nX₁ ... Xₙ', fillcolor='#d5e8d4', color='#82b366')
        prep.node('TokenEmb', 'Embedding\nde Tokens', fillcolor='#dae8fc', color='#6c8ebf')
        prep.node('PosEmb', 'Embedding\nPosicional', fillcolor='#fff2cc', color='#d6b656')
        
        prep.node('AddEmb', '⊕', shape='circle', fillcolor='#f5f5f5', color='#666666', width='0.25', fixedsize='true')
        
        prep.edge('Input', 'TokenEmb')
        prep.edge('TokenEmb', 'AddEmb')
        prep.edge('PosEmb', 'AddEmb')

    # bloco do encoder
    with dot.subgraph(name='cluster_encoder') as encoder:
        encoder.attr(style='dashed,rounded', color='#b85450', label='Encoder', fontname='Helvetica-Bold', fontsize='10')
        encoder.node('MHA', 'Auto-\natenção', fillcolor='#ffe6cc', color='#d79b00')
        encoder.node('AddNorm1', 'Soma &\nNorm', fillcolor='#e1d5e7', color='#9673a6')
        encoder.node('FFN', 'Feed-\nForward', fillcolor='#dae8fc', color='#6c8ebf')
        encoder.node('AddNorm2', 'Soma &\nNorm', fillcolor='#e1d5e7', color='#9673a6')
        
        encoder.edge('MHA', 'AddNorm1', style='bold')
        encoder.edge('AddNorm1', 'FFN', style='bold')
        encoder.edge('FFN', 'AddNorm2', style='bold')

    # parte da saída
    with dot.subgraph(name='cluster_classification') as classif:
        classif.attr(style='dashed', color='#cccccc', label='Saída', fontname='Helvetica-Bold', fontsize='10')
        classif.node('Pool', 'Avg\nPool', fillcolor='#d5e8d4', color='#82b366')
        classif.node('Output', 'Linear\n(+)/(-)', fillcolor='#f8cecc', color='#b85450', shape='ellipse')
        
        classif.edge('Pool', 'Output', style='bold')

    dot.edge('AddEmb', 'MHA', style='bold', label=' (batch, 200, 32)')
    dot.edge('AddNorm2', 'Pool', style='bold')

    # conexões residuais
    dot.edge('AddEmb', 'AddNorm1', style='dashed', color='#b85450', label=' Atalho', constraint='false')
    dot.edge('AddNorm1', 'AddNorm2', style='dashed', color='#b85450', label=' Atalho', constraint='false')

    return dot

diagram = transformer_diagram()
display(diagram)