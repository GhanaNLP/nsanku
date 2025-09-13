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

def create_horizontal_bar_chart(data, title, xlabel, filename, output_dir):
    """Horizontal bar chart sorted lowest → highest (only change)."""
    sorted_data = sorted(data.items(), key=lambda x: x[1])  # reverse=False
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
    """Stacked horizontal bar chart sorted lowest → highest (only change)."""
    if not data_dict:
        return None
    all_sources = sorted({s for model_data in data_dict.values() for s in model_data.keys()})

    model_totals = {model: sum(sources.values()) for model, sources in data_dict.items()}
    sorted_models = sorted(model_totals.items(), key=lambda x: x[1])  # reverse=False
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

def generate_language_specific_reports(results, source_breakdown, output_dir="reports"):
    """Generate individual reports for each language pair with source breakdown"""
    for language_pair, model_results in results.items():
        # Create language-specific directory
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
        
        # Add source breakdown to summary if available
        if language_pair in source_breakdown and source_breakdown[language_pair]:
            summary['source_breakdown'] = source_breakdown[language_pair]
        
        # Save summary as JSON
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
        
    # Prepare data for language performance summary
    language_performance = {}
    
    for language_pair, model_results in results.items():
        # Calculate average performance for this language pair across all models
        scores = list(model_results.values())
        if scores:
            language_performance[language_pair] = np.mean(scores)
    
    # Create language performance chart
    create_horizontal_bar_chart(
        language_performance,
        'Language Translation Performance Across Models (DeepSeek, OpenAI OSS, Llama)',
        'Average Accuracy Score (%)',
        'language_performance',
        output_dir
    )
    
    # Save language performance data to CSV - sort by highest score first
    sorted_languages_for_csv = sorted(language_performance.items(), key=lambda x: x[1], reverse=True)
    language_df = pd.DataFrame({
        'Language Pair': [item[0] for item in sorted_languages_for_csv],
        'Average Score (%)': [item[1] for item in sorted_languages_for_csv]
    })
    language_df.to_csv(os.path.join(output_dir, 'language_performance.csv'), index=False)
    
    return language_performance

def generate_overall_summary(results, source_breakdown, output_dir="reports"):
    """Generate an overall summary across all language pairs"""
    if not results:
        return
        
    # Prepare data for overall summary
    all_models = set()
    for lang_results in results.values():
        all_models.update(lang_results.keys())
    
    # Calculate average performance per model across all languages
    model_performance = {}
    for model in all_models:
        scores = []
        for lang_results in results.values():
            if model in lang_results:
                scores.append(lang_results[model])
        if scores:
            model_performance[model] = np.mean(scores)
    
    # Generate language performance summary
    language_performance = generate_language_performance_summary(results, output_dir)
    
    # Find best performers for summary
    best_model = max(model_performance.items(), key=lambda x: x[1]) if model_performance else ("none", 0)
    best_language = max(language_performance.items(), key=lambda x: x[1]) if language_performance else ("none", 0)
    
    # Create overall summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_language_pairs': len(results),
        'total_models': len(all_models),
        'model_performance': model_performance,
        'language_performance': language_performance,
        'best_overall_model': best_model[0],
        'best_overall_score': best_model[1],
        'best_language_pair': best_language[0],
        'best_language_score': best_language[1],
        'language_pairs': list(results.keys())
    }
    
    # Save overall summary
    with open(os.path.join(output_dir, 'overall_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Create overall performance chart
    if model_performance:
        create_horizontal_bar_chart(
            model_performance,
            'Overall Model Performance Across All Language Pairs',
            'Average Accuracy Score (%)',
            'overall_performance',
            output_dir
        )
    
    return summary

def generate_report(input_dir="output", output_dir="reports"):
    """Main function to generate reports"""
    print("Generating performance reports with Plotly...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect results from all processed files
    results, source_breakdown = collect_results(input_dir)
    
    if not results:
        print("No processed results found. Please run translations first.")
        return
    
    # Generate language-specific reports
    generate_language_specific_reports(results, source_breakdown, output_dir)
    
    # Generate overall summary
    overall_summary = generate_overall_summary(results, source_breakdown, output_dir)
    
    print(f"Reports generated successfully in {output_dir}/")
    print("Both interactive HTML charts and static PNG images have been created!")
    
    if overall_summary:
        print(f"Overall best model: {overall_summary['best_overall_model']} ({overall_summary['best_overall_score']:.2f}%)")
        print(f"Best performing language pair: {overall_summary['best_language_pair']} ({overall_summary['best_language_score']:.2f}%)")
    
    return results, overall_summary
