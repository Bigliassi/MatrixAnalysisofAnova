import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os

# Define electrode positions (Replace with your actual electrode positions)
electrode_positions = {
    'F3': (0, 1),
    'F4': (1, 1),
    'C3': (0, 0),
    'C4': (1, 0),
    'P3': (0, -1),
    'P4': (1, -1),
    'F7': (-1, 1),
    'F8': (2, 1),
    'T7': (-1, 0),
    'T8': (2, 0),
    'P7': (-1, -1),
    'P8': (2, -1),
    'Fz': (0.5, 1.5),
    'Cz': (0.5, 0.5),
    'Pz': (0.5, -0.5),
    'CPz': (0.5, 0),
    'POz': (0.5, -1),
    # Add any other electrodes as needed
}

# Define color mappings for interaction effects
comparison_color_mapping = {
    'hispanic vs non-hispanic in sed': 'blue',
    'hispanic vs non-hispanic in exer': 'green',
    'sed vs exer in hispanic': 'red',
    'sed vs exer in non-hispanic': 'orange',
    # Add more mappings if needed
}

# Define color mappings for group and condition effects
color_mapping = {
    'non-hispanic': 'red',
    'hispanic': 'blue',
    'sed': 'green',
    'exer': 'orange',
    # Add more mappings if needed
}

# Define the order of conditions and groups for standardization
condition_order = ['sed', 'exer']
group_order = ['hispanic', 'non-hispanic']

def standardize_conditions(A, B):
    A_index = condition_order.index(A)
    B_index = condition_order.index(B)
    if A_index <= B_index:
        return A, B
    else:
        return B, A

def standardize_groups(A, B):
    A_index = group_order.index(A)
    B_index = group_order.index(B)
    if A_index <= B_index:
        return A, B
    else:
        return B, A

def get_mean_value(means_df, contrast_type, group_or_condition):
    # Clean group and condition names in means_df
    means_df.index = means_df.index.str.strip().str.lower()
    means_df.columns = means_df.columns.str.strip().str.lower()

    if contrast_type in ['Group', 'Condition * Group']:
        group = group_or_condition.strip().lower()
        if group in means_df.index:
            # Return the mean for the group across all conditions
            mean_value = means_df.loc[group].mean()
            return mean_value
        else:
            print(f"Warning: Group '{group}' not found in means_df index.")
            return None
    elif contrast_type == 'Condition':
        condition = group_or_condition.strip().lower()
        if condition in means_df.columns:
            # Return the mean for the condition across all groups
            mean_value = means_df[condition].mean()
            return mean_value
        else:
            print(f"Warning: Condition '{condition}' not found in means_df columns.")
            return None
    else:
        print(f"Warning: Unknown contrast type '{contrast_type}'.")
        return None

def determine_higher_mean_group(xls, electrode1, electrode2, effect_type):
    dv_col = f"{electrode1}_{electrode2}"
    means_sheet = f'{dv_col}_Means'
    if means_sheet in xls.sheet_names:
        means_df = pd.read_excel(xls, means_sheet)
        # Clean group names and set index
        means_df['Group'] = means_df['Group'].astype(str).str.strip().str.lower()
        means_df.set_index('Group', inplace=True)
        means_df.columns = means_df.columns.str.strip().str.lower()
        means_df = means_df.apply(pd.to_numeric, errors='coerce')

        if effect_type == 'Group':
            # Calculate mean for each group across all conditions
            group_means = means_df.mean(axis=1)
            if 'hispanic' in group_means and 'non-hispanic' in group_means:
                if group_means['hispanic'] > group_means['non-hispanic']:
                    return 'hispanic'
                else:
                    return 'non-hispanic'
            else:
                print(f"Groups 'hispanic' or 'non-hispanic' not found in group_means")
                return None

        elif effect_type == 'Condition':
            # Calculate mean for each condition across all groups
            condition_means = means_df.mean(axis=0)
            if 'sed' in condition_means and 'exer' in condition_means:
                if condition_means['sed'] > condition_means['exer']:
                    return 'sed'
                else:
                    return 'exer'
            else:
                print(f"Conditions 'sed' or 'exer' not found in condition_means")
                return None
    else:
        print(f'Means sheet {means_sheet} not found.')
        return None

def process_excel_file(excel_path):
    xls = pd.ExcelFile(excel_path)
    interaction_effects = []
    group_effects = []
    condition_effects = []

    sheet_names = xls.sheet_names
    print("Excel file sheets:", sheet_names)

    # Loop over sheets ending with '_Mixed_ANOVA' to get main effects
    for sheet_name in sheet_names:
        if sheet_name.endswith('_Mixed_ANOVA'):
            print(f"Processing Mixed ANOVA sheet: {sheet_name}")
            df = pd.read_excel(xls, sheet_name)
            # Ensure 'p-unc' column is numeric
            df['p-unc'] = pd.to_numeric(df['p-unc'], errors='coerce')
            df = df.dropna(subset=['p-unc'])

            # Get the electrode pair from the sheet name
            dv_col = sheet_name.replace('_Mixed_ANOVA', '')
            electrodes = dv_col.split('_')
            if len(electrodes) >= 2:
                electrode1 = electrodes[0]
                electrode2 = electrodes[1]
            else:
                print(f"Could not parse electrode names from sheet name: {sheet_name}")
                continue

            # Check for significant main effects and interactions
            for index, row in df.iterrows():
                source = row['Source'].strip()
                p_value = row['p-unc']
                if p_value < 0.001:
                    if source == 'Interaction':
                        # Process interaction effect
                        interaction_effects.append({
                            'electrode1': electrode1,
                            'electrode2': electrode2,
                            'effect_type': 'Interaction',
                            'p_value': p_value
                        })
                    elif source == 'Group':
                        # Process main effect of group
                        higher_group = determine_higher_mean_group(xls, electrode1, electrode2, 'Group')
                        if higher_group is not None:
                            group_effects.append({
                                'electrode1': electrode1,
                                'electrode2': electrode2,
                                'effect_type': 'Group',
                                'higher': higher_group,
                                'p_value': p_value
                            })
                    elif source == 'Condition':
                        # Process main effect of condition
                        higher_condition = determine_higher_mean_group(xls, electrode1, electrode2, 'Condition')
                        if higher_condition is not None:
                            condition_effects.append({
                                'electrode1': electrode1,
                                'electrode2': electrode2,
                                'effect_type': 'Condition',
                                'higher': higher_condition,
                                'p_value': p_value
                            })
        else:
            continue

    # Now process the interaction effects
    final_interaction_effects = []
    for interaction in interaction_effects:
        electrode1 = interaction['electrode1']
        electrode2 = interaction['electrode2']
        dv_col = f"{electrode1}_{electrode2}"

        posthoc_sheet_name = f"{dv_col}_Posthoc_Comparisons"
        means_sheet_name = f"{dv_col}_Means"

        if posthoc_sheet_name in sheet_names and means_sheet_name in sheet_names:
            print(f"Processing Posthoc Comparisons sheet: {posthoc_sheet_name}")
            df_posthoc = pd.read_excel(xls, posthoc_sheet_name)
            df_posthoc['p-unc'] = pd.to_numeric(df_posthoc['p-unc'], errors='coerce')
            df_posthoc = df_posthoc.dropna(subset=['p-unc'])
            significant_rows = df_posthoc[df_posthoc['p-unc'] < 0.001]

            df_means = pd.read_excel(xls, means_sheet_name)
            df_means['Group'] = df_means['Group'].astype(str).str.strip().str.lower()
            df_means.set_index('Group', inplace=True)
            df_means.columns = df_means.columns.str.strip().str.lower()
            df_means = df_means.apply(pd.to_numeric, errors='coerce')

            groups = df_means.index.tolist()
            conditions = df_means.columns.tolist()

            for idx, row in significant_rows.iterrows():
                contrast = row['Contrast'].strip()
                A = row['A'].strip().lower()
                B = row['B'].strip().lower()
                p_value = row['p-unc']

                if contrast == 'Group':
                    for condition in conditions:
                        mean_A = df_means.loc[A, condition]
                        mean_B = df_means.loc[B, condition]
                        higher = A if mean_A > mean_B else B
                        # Standardize group comparison
                        group_A, group_B = standardize_groups(A, B)
                        comparison = f"{group_A} vs {group_B} in {condition}"
                        final_interaction_effects.append({
                            'electrode1': electrode1,
                            'electrode2': electrode2,
                            'effect_type': contrast,
                            'comparison': comparison,
                            'higher': higher,
                            'p_value': p_value
                        })
                elif contrast == 'Condition':
                    for group in groups:
                        mean_A = df_means.loc[group, A]
                        mean_B = df_means.loc[group, B]
                        higher = A if mean_A > mean_B else B
                        # Standardize condition comparison
                        cond_A, cond_B = standardize_conditions(A, B)
                        comparison = f"{cond_A} vs {cond_B} in {group}"
                        final_interaction_effects.append({
                            'electrode1': electrode1,
                            'electrode2': electrode2,
                            'effect_type': contrast,
                            'comparison': comparison,
                            'higher': higher,
                            'p_value': p_value
                        })
                elif contrast == 'Condition * Group':
                    # Handle interactions if needed
                    pass  # Add handling if required
                else:
                    print(f"Warning: Unknown contrast '{contrast}'")
                    continue
        else:
            print(f"Posthoc sheet or Means sheet not found for {dv_col}")

    return group_effects, condition_effects, final_interaction_effects

def plot_connectivity_graph(effect_list, title, filename):
    G = nx.DiGraph()
    edge_colors = []
    labels = set()
    
    for effect in effect_list:
        electrode1 = effect['electrode1']
        electrode2 = effect['electrode2']
        higher = effect.get('higher')
        if higher is None:
            continue  # Skip this edge
        color = color_mapping.get(higher)
        if color is None:
            print(f"Warning: Color not found for higher value '{higher}'")
            continue  # Skip this edge
        G.add_edge(electrode1, electrode2, color=color)
        edge_colors.append(color)
        labels.add(higher)
    
    pos = electrode_positions  # Should be a dictionary of electrode positions
    plt.figure(figsize=(10, 10))
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=500)
    
    # Draw edges with arrows
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, arrows=True, arrowstyle='->', arrowsize=20)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos)
    
    # Create legend
    from matplotlib.lines import Line2D
    legend_elements = []
    for label in labels:
        if label in color_mapping:
            legend_elements.append(Line2D([0], [0], color=color_mapping[label], lw=2, label=label))
    plt.legend(handles=legend_elements)
    
    plt.title(title)
    plt.savefig(filename)
    plt.close()

def plot_interaction_connectivity_graph(interaction_effects, title, filename):
    G = nx.DiGraph()
    edge_colors = []
    labels = set()
    
    for effect in interaction_effects:
        electrode1 = effect['electrode1']
        electrode2 = effect['electrode2']
        comparison = effect.get('comparison', '')
        color = comparison_color_mapping.get(comparison)
        if color is None:
            print(f"Warning: Color not found for comparison '{comparison}'")
            continue  # Skip this edge
        G.add_edge(electrode1, electrode2, color=color)
        edge_colors.append(color)
        labels.add(comparison)
    
    pos = electrode_positions  # Should be a dictionary of electrode positions
    plt.figure(figsize=(10, 10))
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=500)
    
    # Draw edges with arrows
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, arrows=True, arrowstyle='->', arrowsize=20)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos)
    
    # Create legend
    from matplotlib.lines import Line2D
    legend_elements = []
    for label in labels:
        if label in comparison_color_mapping:
            legend_elements.append(Line2D([0], [0], color=comparison_color_mapping[label], lw=2, label=label))
    plt.legend(handles=legend_elements)
    
    plt.title(title)
    plt.savefig(filename)
    plt.close()

if __name__ == '__main__':
    # Path to your Excel file with ANOVA results
    excel_file_path = r'RM_ANOVA_results_with_interactions.xlsx'
    
    # Process the Excel file to extract effects
    group_effects_list, condition_effects_list, interaction_effects_list = process_excel_file(excel_file_path)
    
    # Plot Group Effects (Figure 1)
    plot_connectivity_graph(group_effects_list, "Group Effects Connectivity", "group_effects.png")
    
    # Plot Condition Effects (Figure 2)
    plot_connectivity_graph(condition_effects_list, "Condition Effects Connectivity", "condition_effects.png")
    
    # Plot Interaction Effects (Figure 3)
    plot_interaction_connectivity_graph(interaction_effects_list, "Interaction Effects Connectivity", "interaction_effects_final.png")
