import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from pathlib import Path
import json
import numpy as np
from datetime import datetime
import re

pio.templates.default = "plotly_white"

# Load language mapping from CSV
def load_language_mapping():
    """Load language mapping from CSV file in utils folder"""
    mapping_path = os.path.join("utils", "language_mapping.csv")
    language_mapping = {}
    
    try:
        if os.path.exists(mapping_path):
            df = pd.read_csv(mapping_path)
            for _, row in df.iterrows():
                language_mapping[row['language_code']] = row['language_name']
            print(f"Loaded language mapping for {len(language_mapping)} languages")
        else:
            print(f"Warning: Language mapping file not found at {mapping_path}")
    except Exception as e:
        print(f"Error loading language mapping: {e}")
    
    return language_mapping

# Load language mapping at module level
LANGUAGE_MAPPING = load_language_mapping()

def get_language_name(lang_code):
    """Get language name from code using the loaded mapping"""
    return LANGUAGE_MAPPING.get(lang_code, lang_code)

def get_language_display_name(language_pair):
    """Convert language pair code to display name using the SOURCE language"""
    if '-' not in language_pair:
        return get_language_name(language_pair)
    
    source_lang, target_lang = language_pair.split('-', 1)
    source_name = get_language_name(source_lang)
    
    return source_name if source_name != source_lang else language_pair

def get_available_recipes(recipes_dir="recipes"):
    recipes = []
    for file in os.listdir(recipes_dir):
        if file.endswith(".py") and file != "__init__.py":
            recipes.append(file[:-3])
    return recipes

def extract_recipe_name_from_filename(filename, available_recipes):
    name_without_ext = os.path.splitext(filename)[0]
    for recipe in available_recipes:
        if f"_{recipe}" in name_without_ext:
            return recipe
    match = re.search(r'_([^_]+)$', name_without_ext)
    if match:
        return match.group(1)
    return "unknown_recipe"

def combine_all_datasets(input_dir="output"):
    """
    Combine all CSV files from all directories into one main dataset
    """
    all_data = []
    recipes = get_available_recipes()
    
    print("Combining all datasets...")
    
    for root, _, files in os.walk(input_dir):
        for file in files:
            if not file.endswith(".csv"):
                continue
            
            folder = os.path.basename(root)
            if "-" not in folder:
                continue
            
            src, tgt = folder.split("-", 1)
            recipe = extract_recipe_name_from_filename(file, recipes)
            
            try:
                df = pd.read_csv(os.path.join(root, file))
                if "similarity_score" not in df.columns:
                    continue
                
                # Add metadata columns
                df['language_pair'] = f"{src}-{tgt}"
                df['source_lang'] = src
                df['target_lang'] = tgt
                df['model'] = recipe
                df['file_path'] = os.path.join(root, file)
                
                # Convert similarity score to percentage
                df['similarity_score_pct'] = df['similarity_score'] * 100
                
                all_data.append(df)
                
            except Exception as e:
                print(f"Error reading {file}: {e}")
                continue
    
    if not all_data:
        print("No data files found!")
        return pd.DataFrame()
    
    # Combine all dataframes
    combined_df = pd.concat(all_data, ignore_index=True)
    
    print(f"Combined dataset shape: {combined_df.shape}")
    print(f"Language pairs: {combined_df['language_pair'].nunique()}")
    print(f"Models: {combined_df['model'].nunique()}")
    
    return combined_df

def calculate_metrics(df):
    """
    Calculate performance metrics for quadrant analysis
    """
    if df.empty:
        return pd.DataFrame()
    
    # Calculate metrics by model and language pair
    metrics = []
    
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        
        # Overall metrics
        avg_score = model_data['similarity_score_pct'].mean()
        consistency = 100 - model_data['similarity_score_pct'].std()  # Higher std = lower consistency
        data_points = len(model_data)
        coverage = model_data['language_pair'].nunique()  # Number of language pairs covered
        
        # Performance across language pairs
        lang_performance = model_data.groupby('language_pair')['similarity_score_pct'].mean()
        versatility = 100 - lang_performance.std()  # Lower std across languages = higher versatility
        
        # Source breakdown if available
        if 'source' in model_data.columns:
            source_performance = model_data.groupby('source')['similarity_score_pct'].mean()
            source_consistency = 100 - source_performance.std()
        else:
            source_consistency = consistency
        
        metrics.append({
            'model': model,
            'avg_score': avg_score,
            'consistency': max(0, consistency),  # Ensure non-negative
            'versatility': max(0, versatility),  # Ensure non-negative
            'source_consistency': max(0, source_consistency),
            'coverage': coverage,
            'data_points': data_points,
            'max_score': model_data['similarity_score_pct'].max(),
            'min_score': model_data['similarity_score_pct'].min()
        })
    
    return pd.DataFrame(metrics)

def create_enhanced_quadrant_chart(metrics_df, x_metric, y_metric, title, filename, outdir, 
                                 id_column, size_metric=None, color_metric=None):
    """
    Create an enhanced quadrant chart with clean label-only visualization
    """
    if metrics_df.empty:
        return None
    
    # Calculate quadrant lines (medians)
    x_line = metrics_df[x_metric].median()
    y_line = metrics_df[y_metric].median()
    
    # Create scatter plot
    fig = go.Figure()
    
    # Calculate axis ranges with some padding
    x_min, x_max = metrics_df[x_metric].min(), metrics_df[x_metric].max()
    y_min, y_max = metrics_df[y_metric].min(), metrics_df[y_metric].max()
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_padding = x_range * 0.05
    y_padding = y_range * 0.05
    
    # Add quadrant background regions with labels positioned on axis lines
    regions = [
        {
            'x0': x_min - x_padding, 'x1': x_line,
            'y0': y_line, 'y1': y_max + y_padding,
            'color': 'lightblue', 
            'label': 'Consistent\nBut Average',
            'label_x': x_min + (x_line - x_min) * 0.5,
            'label_y': y_max + y_padding * 0.8,
            'align': 'center'
        },
        {
            'x0': x_line, 'x1': x_max + x_padding,
            'y0': y_line, 'y1': y_max + y_padding,
            'color': 'lightgreen', 
            'label': 'LEADERS\n(High Performance & Consistency)',
            'label_x': x_line + (x_max - x_line) * 0.5,
            'label_y': y_max + y_padding * 0.8,
            'align': 'center'
        },
        {
            'x0': x_min - x_padding, 'x1': x_line,
            'y0': y_min - y_padding, 'y1': y_line,
            'color': 'lightcoral', 
            'label': 'NEEDS IMPROVEMENT\n(Low Performance & Consistency)',
            'label_x': x_min + (x_line - x_min) * 0.5,
            'label_y': y_min - y_padding * 0.8,
            'align': 'center'
        },
        {
            'x0': x_line, 'x1': x_max + x_padding,
            'y0': y_min - y_padding, 'y1': y_line,
            'color': 'lightyellow', 
            'label': 'High Performance\nBut Inconsistent',
            'label_x': x_line + (x_max - x_line) * 0.5,
            'label_y': y_min - y_padding * 0.8,
            'align': 'center'
        }
    ]
    
    for region in regions:
        fig.add_shape(
            type="rect",
            x0=region['x0'], x1=region['x1'],
            y0=region['y0'], y1=region['y1'],
            fillcolor=region['color'], opacity=0.2, line_width=0
        )
        
        # Add region labels positioned on axis lines
        label_color = 'darkgreen' if 'LEADERS' in region['label'] else 'darkred' if 'NEEDS' in region['label'] else 'gray'
        font_weight = 'bold' if 'LEADERS' in region['label'] else 'normal'
        
        fig.add_annotation(
            x=region['label_x'], 
            y=region['label_y'],
            text=region['label'], 
            showarrow=False,
            xanchor=region['align'],
            font=dict(size=12, color=label_color, weight=font_weight), 
            opacity=0.8,
            bgcolor="white",
            bordercolor=label_color,
            borderwidth=1,
            borderpad=4
        )
    
    # Add quadrant lines
    fig.add_vline(x=x_line, line_dash="dash", line_color="gray", line_width=2)
    fig.add_hline(y=y_line, line_dash="dash", line_color="gray", line_width=2)
    
    # Use a force-directed approach to position data labels
    label_positions = {}
    
    # First pass: initial positions
    for i, row in metrics_df.iterrows():
        x_val = row[x_metric]
        y_val = row[y_metric]
        
        label_positions[i] = {
            'x': x_val,
            'y': y_val,
            'original_x': x_val,
            'original_y': y_val
        }
    
    # Second pass: resolve overlaps with a simple force-directed algorithm
    max_iterations = 100
    for iteration in range(max_iterations):
        moved = False
        for i, pos_i in label_positions.items():
            for j, pos_j in label_positions.items():
                if i >= j:
                    continue
                
                # Calculate distance between labels
                dx = pos_j['x'] - pos_i['x']
                dy = pos_j['y'] - pos_i['y']
                distance = (dx**2 + dy**2)**0.5
                
                # If labels are too close, push them apart
                min_distance = x_range * 0.08
                if distance < min_distance and distance > 0:
                    force = (min_distance - distance) / distance
                    
                    label_positions[i]['x'] -= dx * force * 0.5
                    label_positions[i]['y'] -= dy * force * 0.5
                    label_positions[j]['x'] += dx * force * 0.5
                    label_positions[j]['y'] += dy * force * 0.5
                    
                    moved = True
        
        # Also pull labels toward their original positions
        for i, pos in label_positions.items():
            original_dx = pos['original_x'] - pos['x']
            original_dy = pos['original_y'] - pos['y']
            original_dist = (original_dx**2 + original_dy**2)**0.5
            
            if original_dist > x_range * 0.15:
                label_positions[i]['x'] += original_dx * 0.1
                label_positions[i]['y'] += original_dy * 0.1
                moved = True
        
        if not moved:
            break
    
    # Add only text labels (no points or connecting lines)
    for i, row in metrics_df.iterrows():
        label_text = row[id_column]
        label_pos = label_positions[i]
        
        fig.add_trace(go.Scatter(
            x=[label_pos['x']],
            y=[label_pos['y']],
            mode='text',
            text=[label_text],
            textposition="middle center",
            textfont=dict(size=12, color="black", family="Arial"),
            showlegend=False,
            hoverinfo='text',
            hovertext=(
                f"<b>{label_text}</b><br>" +
                f"{x_metric.replace('_', ' ').title()}: {row[x_metric]:.2f}<br>" +
                f"{y_metric.replace('_', ' ').title()}: {row[y_metric]:.2f}<br>" +
                (f"{size_metric.replace('_', ' ').title()}: {row[size_metric]}<br>" if size_metric else "") +
                (f"{color_metric.replace('_', ' ').title()}: {row[color_metric]:.2f}<br>" if color_metric else "")
            )
        ))
    
    # Adjust layout with extra padding for top and bottom quadrant labels
    chart_height = 900
    chart_width = 1000
    
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=18)),
        xaxis=dict(
            title=x_metric.replace('_', ' ').title(),
            gridcolor="lightgray", gridwidth=1, showgrid=True,
            range=[x_min - x_padding, x_max + x_padding]
        ),
        yaxis=dict(
            title=y_metric.replace('_', ' ').title(),
            gridcolor="lightgray", gridwidth=1, showgrid=True,
            range=[y_min - y_padding * 2, y_max + y_padding * 2]
        ),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=100, r=100, t=120, b=120),
        height=chart_height, width=chart_width
    )
    
    # Save chart
    fig.write_html(os.path.join(outdir, f"{filename}.html"))
    fig.write_image(os.path.join(outdir, f"{filename}.png"), width=chart_width, height=chart_height)
    
    return fig

def create_language_quadrant(df, title, filename, outdir):
    """
    Create quadrant chart for language pair performance with proper language names
    """
    if df.empty:
        return None
    
    # Calculate metrics by language pair
    lang_metrics = []
    for lp in df['language_pair'].unique():
        lp_data = df[df['language_pair'] == lp]
        
        avg_score = lp_data['similarity_score_pct'].mean()
        consistency = 100 - lp_data['similarity_score_pct'].std()
        model_count = lp_data['model'].nunique()
        
        # Get display name for the language pair using SOURCE language
        display_name = get_language_display_name(lp)
        
        lang_metrics.append({
            'language_pair': lp,
            'display_name': display_name,
            'avg_score': avg_score,
            'consistency': max(0, consistency),
            'model_coverage': model_count
        })
    
    lang_df = pd.DataFrame(lang_metrics)
    
    if lang_df.empty:
        return None
    
    return create_enhanced_quadrant_chart(
        lang_df, 'avg_score', 'consistency', title, filename, outdir,
        id_column='display_name', size_metric='model_coverage', color_metric='avg_score'
    )

def generate_quadrant_reports(combined_df, outdir="reports"):
    """
    Generate quadrant chart reports
    """
    if combined_df.empty:
        print("No data available for quadrant reporting")
        return
    
    os.makedirs(outdir, exist_ok=True)
    
    # Save combined dataset
    combined_df.to_csv(os.path.join(outdir, "combined_dataset.csv"), index=False)
    print(f"Combined dataset saved with {len(combined_df)} records")
    
    # Calculate metrics for models
    metrics_df = calculate_metrics(combined_df)
    if not metrics_df.empty:
        metrics_df.to_csv(os.path.join(outdir, "model_metrics.csv"), index=False)
    
    # Generate quadrant charts
    # 1. Model Performance vs Consistency Quadrant
    create_enhanced_quadrant_chart(
        metrics_df, 'avg_score', 'consistency',
        'Model Performance vs Consistency Analysis',
        'model_performance_quadrant', outdir,
        id_column='model', size_metric='coverage', color_metric='versatility'
    )
    
    # 2. Language Performance Quadrant with proper names
    create_language_quadrant(
        combined_df,
        'Language Performance vs Consistency Analysis',
        'language_performance_quadrant', outdir
    )
    
    print("Quadrant reports generated successfully!")

def create_horizontal_bar_chart(data, title, xlabel, filename, output_dir):
    """Horizontal bar chart sorted lowest → highest"""
    sorted_data = sorted(data.items(), key=lambda x: x[1])
    labels = [item[0] for item in sorted_data]
    values = [item[1] for item in sorted_data]

    fig = go.Figure()
    colors = px.colors.qualitative.Set3[:len(labels)]
    if len(labels) > len(colors):
        colors *= (len(labels) // len(colors) + 1)

    fig.add_trace(go.Bar(
        x=values, y=labels, orientation='h',
        marker=dict(color=colors[:len(labels)],
                    line=dict(color='rgba(50, 50, 50, 0.8)', width=1)),
        text=[f'{val:.2f}%' for val in values],
        textposition='outside', textfont=dict(size=12, color='black')
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=18, family="Arial, sans-serif"), x=0.5),
        xaxis=dict(title=xlabel, showgrid=True, gridcolor='lightgray', gridwidth=1, title_font=dict(size=14)),
        yaxis=dict(title='', showgrid=False, categoryorder='array', categoryarray=labels),
        plot_bgcolor='white', paper_bgcolor='white',
        font=dict(family="Arial, sans-serif"),
        margin=dict(l=150, r=80, t=80, b=60),
        height=max(400, len(labels) * 50 + 100)
    )

    fig.write_html(os.path.join(output_dir, f"{filename}.html"))
    fig.write_image(os.path.join(output_dir, f"{filename}.png"),
                    width=1200, height=max(400, len(labels) * 50 + 100))
    return fig

def create_stacked_bar_chart(data_dict, title, xlabel, filename, output_dir):
    """Stacked horizontal bar chart sorted lowest → highest"""
    if not data_dict:
        return None
    all_sources = sorted({s for model_data in data_dict.values() for s in model_data.keys()})

    model_totals = {model: sum(sources.values()) for model, sources in data_dict.items()}
    sorted_models = sorted(model_totals.items(), key=lambda x: x[1])
    model_order = [m for m, _ in sorted_models]

    fig = go.Figure()
    colors = px.colors.qualitative.Set3
    if len(all_sources) > len(colors):
        colors *= (len(all_sources) // len(colors) + 1)

    for i, source in enumerate(all_sources):
        vals = [data_dict.get(model, {}).get(source, 0) for model in model_order]
        fig.add_trace(go.Bar(
            name=source, x=vals, y=model_order, orientation='h',
            marker=dict(color=colors[i % len(colors)]),
            text=[f'{v:.1f}%' if v > 0 else '' for v in vals],
            textposition='inside'
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=18, family="Arial, sans-serif"), x=0.5),
        xaxis=dict(title=xlabel, showgrid=True, gridcolor='lightgray', gridwidth=1, title_font=dict(size=14)),
        yaxis=dict(title='', showgrid=False, categoryorder='array', categoryarray=model_order),
        barmode='stack',
        plot_bgcolor='white', paper_bgcolor='white',
        font=dict(family="Arial, sans-serif"),
        margin=dict(l=150, r=80, t=80, b=60),
        height=max(400, len(model_order) * 60 + 150),
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
    )

    fig.write_html(os.path.join(output_dir, f"{filename}.html"))
    fig.write_image(os.path.join(output_dir, f"{filename}.png"),
                    width=1400, height=max(400, len(model_order) * 60 + 150))
    return fig

def collect_results(input_dir="output"):
    results, source_breakdown = {}, {}
    available_recipes = get_available_recipes()

    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".csv"):
                folder_name = os.path.basename(root)
                if '-' in folder_name:
                    source_lang, target_lang = folder_name.split('-', 1)
                    recipe_name = extract_recipe_name_from_filename(file, available_recipes)
                    try:
                        df = pd.read_csv(os.path.join(root, file))
                        if 'similarity_score' in df.columns:
                            avg_score = df['similarity_score'].mean()
                            results.setdefault(f"{source_lang}-{target_lang}", {})[recipe_name] = avg_score * 100
                            if 'source' in df.columns:
                                source_breakdown.setdefault(f"{source_lang}-{target_lang}", {})
                                source_breakdown[f"{source_lang}-{target_lang}"].setdefault(recipe_name, {})
                                for source, group in df.groupby('source'):
                                    source_breakdown[f"{source_lang}-{target_lang}"][recipe_name][source] = \
                                        group['similarity_score'].mean() * 100
                    except Exception as e:
                        print(f"Error reading {file}: {e}")
                        continue
    return results, source_breakdown

def generate_language_specific_reports(results, source_breakdown, output_dir="reports"):
    """Generate individual reports for each language pair with source breakdown"""
    for language_pair, model_results in results.items():
        lang_output_dir = os.path.join(output_dir, language_pair)
        os.makedirs(lang_output_dir, exist_ok=True)
        
        if not model_results:
            print(f"No model results found for {language_pair}")
            continue
        
        # Generate language-specific horizontal bar chart (overall)
        create_horizontal_bar_chart(
            model_results,
            f'Translation Quality for {language_pair}',
            'Similarity Score (%)',
            'performance_comparison',
            lang_output_dir
        )
        
        # Generate stacked bar chart by source if we have source breakdown data
        if language_pair in source_breakdown and source_breakdown[language_pair]:
            create_stacked_bar_chart(
                source_breakdown[language_pair],
                f'Translation Quality by Source for {language_pair}',
                'Similarity Score (%)',
                'source_breakdown',
                lang_output_dir
            )
        
        # Generate language-specific CSV report - sort by highest score first
        sorted_models_for_csv = sorted(model_results.items(), key=lambda x: x[1], reverse=True)
        report_data = []
        for model, score in sorted_models_for_csv:
            report_data.append({
                'Model': model,
                'Similarity Score (%)': f"{score:.2f}%",
                'Raw Score': score
            })
        
        report_df = pd.DataFrame(report_data)
        report_df.to_csv(os.path.join(lang_output_dir, 'detailed_report.csv'), index=False)
        
        # Generate source breakdown CSV if available
        if language_pair in source_breakdown and source_breakdown[language_pair]:
            source_report_data = []
            for model, sources in source_breakdown[language_pair].items():
                for source, score in sources.items():
                    source_report_data.append({
                        'Model': model,
                        'Source': source,
                        'Similarity Score (%)': f"{score:.2f}%",
                        'Raw Score': score
                    })
            
            source_report_df = pd.DataFrame(source_report_data)
            source_report_df.to_csv(os.path.join(lang_output_dir, 'source_breakdown.csv'), index=False)
        
        # Generate language-specific summary
        best_model = max(model_results.items(), key=lambda x: x[1])
        summary = {
            'language_pair': language_pair,
            'timestamp': datetime.now().isoformat(),
            'models': model_results,
            'average_score': np.mean(list(model_results.values())) if model_results else 0,
            'best_model': best_model[0] if model_results else "none",
            'best_score': best_model[1] if model_results else 0
        }
        
        if language_pair in source_breakdown and source_breakdown[language_pair]:
            summary['source_breakdown'] = source_breakdown[language_pair]
        
        with open(os.path.join(lang_output_dir, 'summary_report.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Generated report for {language_pair} in {lang_output_dir}/")
        
        # Create a simple text summary
        with open(os.path.join(lang_output_dir, 'summary.txt'), 'w') as f:
            f.write(f"Translation Benchmark Results for {language_pair}\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("\nModel Performance:\n")
            for model, score in sorted_models_for_csv:
                f.write(f"{model}: {score:.2f}%\n")
            f.write(f"\nBest Model: {summary['best_model']} ({summary['best_score']:.2f}%)\n")
            f.write(f"Average Score: {summary['average_score']:.2f}%\n")

def generate_language_performance_summary(results, output_dir="reports"):
    """Generate a summary of how languages cumulatively performed across models"""
    if not results:
        return {}
        
    language_performance = {}
    
    for language_pair, model_results in results.items():
        scores = list(model_results.values())
        if scores:
            # Use the source language name for display
            display_name = get_language_display_name(language_pair)
            language_performance[display_name] = np.mean(scores)
    
    # Create language performance chart
    create_horizontal_bar_chart(
        language_performance,
        'Language Translation Performance Across Models',
        'Average Accuracy Score (%)',
        'language_performance',
        output_dir
    )
    
    # Save language performance data to CSV - sort by highest score first
    sorted_languages_for_csv = sorted(language_performance.items(), key=lambda x: x[1], reverse=True)
    language_df = pd.DataFrame({
        'Language': [item[0] for item in sorted_languages_for_csv],
        'Average Score (%)': [item[1] for item in sorted_languages_for_csv]
    })
    language_df.to_csv(os.path.join(output_dir, 'language_performance.csv'), index=False)
    
    return language_performance

def generate_overall_summary(results, source_breakdown, output_dir="reports"):
    """Generate an overall summary across all language pairs"""
    if not results:
        return
        
    all_models = set()
    for lang_results in results.values():
        all_models.update(lang_results.keys())
    
    model_performance = {}
    for model in all_models:
        scores = []
        for lang_results in results.values():
            if model in lang_results:
                scores.append(lang_results[model])
        if scores:
            model_performance[model] = np.mean(scores)
    
    language_performance = generate_language_performance_summary(results, output_dir)
    
    best_model = max(model_performance.items(), key=lambda x: x[1]) if model_performance else ("none", 0)
    best_language = max(language_performance.items(), key=lambda x: x[1]) if language_performance else ("none", 0)
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_language_pairs': len(results),
        'total_models': len(all_models),
        'model_performance': model_performance,
        'language_performance': language_performance,
        'best_overall_model': best_model[0],
        'best_overall_score': best_model[1],
        'best_language': best_language[0],
        'best_language_score': best_language[1],
        'language_pairs': list(results.keys())
    }
    
    with open(os.path.join(output_dir, 'overall_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    if model_performance:
        create_horizontal_bar_chart(
            model_performance,
            'Overall Model Performance Across All Languages',
            'Average Accuracy Score (%)',
            'overall_performance',
            output_dir
        )
    
    return summary

def generate_report(input_dir="output", output_dir="reports"):
    """Main function to generate reports with quadrant analysis"""
    print("Generating performance reports with quadrant analysis...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Combine all datasets for quadrant analysis
    combined_df = combine_all_datasets(input_dir)
    
    # Step 2: Generate quadrant reports if we have data
    if not combined_df.empty:
        generate_quadrant_reports(combined_df, output_dir)
    else:
        print("No combined data found for quadrant analysis.")
    
    # Step 3: Collect results for traditional reporting (backward compatibility)
    results, source_breakdown = collect_results(input_dir)
    
    if not results:
        print("No processed results found. Please run translations first.")
        return
    
    # Step 4: Generate traditional reports
    generate_language_specific_reports(results, source_breakdown, output_dir)
    overall_summary = generate_overall_summary(results, source_breakdown, output_dir)
    
    print(f"Reports generated successfully in {output_dir}/")
    print("Both traditional reports and quadrant analysis have been created!")
    
    if overall_summary:
        print(f"Overall best model: {overall_summary['best_overall_model']} ({overall_summary['best_overall_score']:.2f}%)")
        print(f"Best performing language: {overall_summary['best_language']} ({overall_summary['best_language_score']:.2f}%)")
    
    return results, overall_summary

if __name__ == "__main__":
    generate_report()
