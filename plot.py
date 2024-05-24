import pandas as pd
import plotly.express as px

log_file_path = 'gridsearch.log'

models = []
train_mae_list = []
test_mae_list = []

with open(log_file_path, 'r') as file:
    lines = file.readlines()
    model_name = None
    for line in lines:
        if "Model Performance:" in line:
            model_name = line.split()[0]
        elif "Train set: R2=" in line:
            train_mae = float(line.split(',')[1].split('=')[1].strip())
        elif "Test set: R2 =" in line:
            test_mae = float(line.split(',')[1].split('=')[1].strip())
            if model_name is not None:  # Check if model_name is set
                models.append(model_name)
                train_mae_list.append(train_mae)
                test_mae_list.append(test_mae)
                model_name = None  # Reset model_name after use

# Create a DataFrame from the parsed data
df = pd.DataFrame({
    'Model': models,
    'Train MAE': train_mae_list,
    'Test MAE': test_mae_list
})

# Step 2: Create a Plot with Plotly
fig = px.bar(df, x='Model', y=['Train MAE', 'Test MAE'],
             barmode='group',
             labels={'value': 'MAE', 'variable': 'Dataset'},
             height=600)

fig.update_layout(
    xaxis_title='Model',
    yaxis_title='Mean Absolute Errors (kcal/mol)',
    legend_title='Dataset',
    font=dict(
        family="Arial, sans-serif",
        size=14,
        color="RebeccaPurple"
    ),
    xaxis=dict(
        title='Model',
        titlefont=dict(size=18, color='black'),
        tickfont=dict(size=14, color='black'),
        showline=True,
        linewidth=2,
        linecolor='black'
    ),
    yaxis=dict(
        title='MAE',
        titlefont=dict(size=18, color='black'),
        tickfont=dict(size=14, color='black'),
        showline=True,
        linewidth=2,
        linecolor='black'
    )
)

fig.show()
