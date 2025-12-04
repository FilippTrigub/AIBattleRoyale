#!/usr/bin/env python3
"""
MongoDB Data Visualization Script
Generates comprehensive PNG charts from the dummy_data collection
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

print("Loading data from JSON file...")
# Load the exported JSON data
with open('/tmp/dummy_data.json', 'r') as f:
    data = [json.loads(line) for line in f]

print(f"Loaded {len(data)} records")

# Convert to DataFrame for easier analysis
df = pd.DataFrame(data)

# Parse dates
df['createdAt'] = pd.to_datetime(df['createdAt'].apply(lambda x: x['$date']))
df['metadata.lastLogin'] = pd.to_datetime(df['metadata'].apply(lambda x: x['lastLogin']['$date']))

# Extract nested fields
df['loginCount'] = df['metadata'].apply(lambda x: x['loginCount'])
df['theme'] = df['metadata'].apply(lambda x: x['preferences']['theme'])
df['language'] = df['metadata'].apply(lambda x: x['preferences']['language'])
df['notifications'] = df['metadata'].apply(lambda x: x['preferences']['notifications'])
df['city'] = df['address'].apply(lambda x: x['city'])
df['state'] = df['address'].apply(lambda x: x['state'])
df['country'] = df['address'].apply(lambda x: x['country'])

# Extract tag counts
df['tag_count'] = df['tags'].apply(len)

print("Generating visualizations...")

# 1. Age Distribution
plt.figure(figsize=(12, 8))
plt.hist(df['age'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
plt.title('Age Distribution of Users', fontsize=16, fontweight='bold')
plt.xlabel('Age', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('/vercel/sandbox/age_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved: age_distribution.png")
plt.close()

# 2. Score Distribution
plt.figure(figsize=(12, 8))
plt.hist(df['score'], bins=50, color='coral', edgecolor='black', alpha=0.7)
plt.title('Score Distribution', fontsize=16, fontweight='bold')
plt.xlabel('Score', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('/vercel/sandbox/score_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved: score_distribution.png")
plt.close()

# 3. Active vs Inactive Users
plt.figure(figsize=(10, 8))
active_counts = df['isActive'].value_counts()
colors = ['#2ecc71', '#e74c3c']
plt.pie(active_counts, labels=['Active', 'Inactive'], autopct='%1.1f%%', 
        colors=colors, startangle=90, textprops={'fontsize': 14})
plt.title('User Activity Status', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('/vercel/sandbox/activity_status.png', dpi=300, bbox_inches='tight')
print("✓ Saved: activity_status.png")
plt.close()

# 4. Theme Preferences
plt.figure(figsize=(10, 8))
theme_counts = df['theme'].value_counts()
colors_theme = ['#34495e', '#ecf0f1']
plt.pie(theme_counts, labels=theme_counts.index, autopct='%1.1f%%', 
        colors=colors_theme, startangle=90, textprops={'fontsize': 14})
plt.title('Theme Preferences (Dark vs Light)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('/vercel/sandbox/theme_preferences.png', dpi=300, bbox_inches='tight')
print("✓ Saved: theme_preferences.png")
plt.close()

# 5. Language Distribution
plt.figure(figsize=(12, 8))
language_counts = df['language'].value_counts()
plt.bar(language_counts.index, language_counts.values, color='teal', edgecolor='black', alpha=0.7)
plt.title('Language Distribution', fontsize=16, fontweight='bold')
plt.xlabel('Language', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=0)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('/vercel/sandbox/language_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved: language_distribution.png")
plt.close()

# 6. Login Count Distribution
plt.figure(figsize=(12, 8))
plt.hist(df['loginCount'], bins=50, color='purple', edgecolor='black', alpha=0.7)
plt.title('Login Count Distribution', fontsize=16, fontweight='bold')
plt.xlabel('Number of Logins', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('/vercel/sandbox/login_count_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved: login_count_distribution.png")
plt.close()

# 7. Age vs Score Scatter Plot
plt.figure(figsize=(12, 8))
plt.scatter(df['age'], df['score'], alpha=0.3, c=df['isActive'].map({True: 'green', False: 'red'}), s=10)
plt.title('Age vs Score Relationship', fontsize=16, fontweight='bold')
plt.xlabel('Age', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.grid(alpha=0.3)
plt.legend(['Inactive', 'Active'], loc='upper right')
plt.tight_layout()
plt.savefig('/vercel/sandbox/age_vs_score.png', dpi=300, bbox_inches='tight')
print("✓ Saved: age_vs_score.png")
plt.close()

# 8. User Registration Over Time
plt.figure(figsize=(14, 8))
df['year_month'] = df['createdAt'].dt.to_period('M')
registrations = df.groupby('year_month').size()
registrations.index = registrations.index.to_timestamp()
plt.plot(registrations.index, registrations.values, linewidth=2, color='dodgerblue', marker='o', markersize=4)
plt.title('User Registrations Over Time', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Number of Registrations', fontsize=12)
plt.xticks(rotation=45)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('/vercel/sandbox/registrations_over_time.png', dpi=300, bbox_inches='tight')
print("✓ Saved: registrations_over_time.png")
plt.close()

# 9. Tag Count Distribution
plt.figure(figsize=(12, 8))
tag_count_dist = df['tag_count'].value_counts().sort_index()
plt.bar(tag_count_dist.index, tag_count_dist.values, color='orange', edgecolor='black', alpha=0.7)
plt.title('Distribution of Tag Counts per User', fontsize=16, fontweight='bold')
plt.xlabel('Number of Tags', fontsize=12)
plt.ylabel('Number of Users', fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('/vercel/sandbox/tag_count_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved: tag_count_distribution.png")
plt.close()

# 10. Notification Preferences
plt.figure(figsize=(10, 8))
notif_counts = df['notifications'].value_counts()
colors_notif = ['#3498db', '#95a5a6']
plt.pie(notif_counts, labels=['Enabled', 'Disabled'], autopct='%1.1f%%', 
        colors=colors_notif, startangle=90, textprops={'fontsize': 14})
plt.title('Notification Preferences', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('/vercel/sandbox/notification_preferences.png', dpi=300, bbox_inches='tight')
print("✓ Saved: notification_preferences.png")
plt.close()

# 11. Top 20 States by User Count
plt.figure(figsize=(14, 8))
top_states = df['state'].value_counts().head(20)
plt.barh(range(len(top_states)), top_states.values, color='steelblue', edgecolor='black', alpha=0.7)
plt.yticks(range(len(top_states)), top_states.index)
plt.title('Top 20 States by User Count', fontsize=16, fontweight='bold')
plt.xlabel('Number of Users', fontsize=12)
plt.ylabel('State', fontsize=12)
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('/vercel/sandbox/top_states.png', dpi=300, bbox_inches='tight')
print("✓ Saved: top_states.png")
plt.close()

# 12. Score by Activity Status (Box Plot)
plt.figure(figsize=(10, 8))
active_scores = df[df['isActive'] == True]['score']
inactive_scores = df[df['isActive'] == False]['score']
plt.boxplot([active_scores, inactive_scores], labels=['Active', 'Inactive'], 
            patch_artist=True, boxprops=dict(facecolor='lightblue', alpha=0.7))
plt.title('Score Distribution by Activity Status', fontsize=16, fontweight='bold')
plt.ylabel('Score', fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('/vercel/sandbox/score_by_activity.png', dpi=300, bbox_inches='tight')
print("✓ Saved: score_by_activity.png")
plt.close()

# 13. Age by Activity Status (Box Plot)
plt.figure(figsize=(10, 8))
active_ages = df[df['isActive'] == True]['age']
inactive_ages = df[df['isActive'] == False]['age']
plt.boxplot([active_ages, inactive_ages], labels=['Active', 'Inactive'], 
            patch_artist=True, boxprops=dict(facecolor='lightgreen', alpha=0.7))
plt.title('Age Distribution by Activity Status', fontsize=16, fontweight='bold')
plt.ylabel('Age', fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('/vercel/sandbox/age_by_activity.png', dpi=300, bbox_inches='tight')
print("✓ Saved: age_by_activity.png")
plt.close()

# 14. Login Count by Theme
plt.figure(figsize=(10, 8))
theme_login = df.groupby('theme')['loginCount'].mean()
plt.bar(theme_login.index, theme_login.values, color=['#34495e', '#ecf0f1'], 
        edgecolor='black', alpha=0.7)
plt.title('Average Login Count by Theme Preference', fontsize=16, fontweight='bold')
plt.xlabel('Theme', fontsize=12)
plt.ylabel('Average Login Count', fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('/vercel/sandbox/login_by_theme.png', dpi=300, bbox_inches='tight')
print("✓ Saved: login_by_theme.png")
plt.close()

# 15. Heatmap: Age Groups vs Language
plt.figure(figsize=(12, 8))
df['age_group'] = pd.cut(df['age'], bins=[0, 25, 35, 45, 55, 65, 100], 
                          labels=['18-25', '26-35', '36-45', '46-55', '56-65', '66+'])
heatmap_data = pd.crosstab(df['age_group'], df['language'])
sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='YlOrRd', cbar_kws={'label': 'Count'})
plt.title('User Count: Age Groups vs Language', fontsize=16, fontweight='bold')
plt.xlabel('Language', fontsize=12)
plt.ylabel('Age Group', fontsize=12)
plt.tight_layout()
plt.savefig('/vercel/sandbox/age_language_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Saved: age_language_heatmap.png")
plt.close()

# 16. Multi-panel Dashboard
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('MongoDB Data Dashboard Overview', fontsize=18, fontweight='bold')

# Panel 1: Age Distribution
axes[0, 0].hist(df['age'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
axes[0, 0].set_title('Age Distribution', fontweight='bold')
axes[0, 0].set_xlabel('Age')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].grid(axis='y', alpha=0.3)

# Panel 2: Activity Status
active_counts = df['isActive'].value_counts()
axes[0, 1].pie(active_counts, labels=['Active', 'Inactive'], autopct='%1.1f%%', 
               colors=['#2ecc71', '#e74c3c'], startangle=90)
axes[0, 1].set_title('Activity Status', fontweight='bold')

# Panel 3: Language Distribution
language_counts = df['language'].value_counts()
axes[1, 0].bar(language_counts.index, language_counts.values, color='teal', 
               edgecolor='black', alpha=0.7)
axes[1, 0].set_title('Language Distribution', fontweight='bold')
axes[1, 0].set_xlabel('Language')
axes[1, 0].set_ylabel('Count')
axes[1, 0].grid(axis='y', alpha=0.3)

# Panel 4: Score Distribution
axes[1, 1].hist(df['score'], bins=30, color='coral', edgecolor='black', alpha=0.7)
axes[1, 1].set_title('Score Distribution', fontweight='bold')
axes[1, 1].set_xlabel('Score')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('/vercel/sandbox/dashboard_overview.png', dpi=300, bbox_inches='tight')
print("✓ Saved: dashboard_overview.png")
plt.close()

print("\n" + "="*60)
print("All visualizations generated successfully!")
print("="*60)
print("\nGenerated PNG files:")
print("  1. age_distribution.png")
print("  2. score_distribution.png")
print("  3. activity_status.png")
print("  4. theme_preferences.png")
print("  5. language_distribution.png")
print("  6. login_count_distribution.png")
print("  7. age_vs_score.png")
print("  8. registrations_over_time.png")
print("  9. tag_count_distribution.png")
print(" 10. notification_preferences.png")
print(" 11. top_states.png")
print(" 12. score_by_activity.png")
print(" 13. age_by_activity.png")
print(" 14. login_by_theme.png")
print(" 15. age_language_heatmap.png")
print(" 16. dashboard_overview.png")
print("="*60)

# Generate summary statistics
print("\nData Summary Statistics:")
print(f"Total Users: {len(df):,}")
print(f"Active Users: {df['isActive'].sum():,} ({df['isActive'].sum()/len(df)*100:.1f}%)")
print(f"Inactive Users: {(~df['isActive']).sum():,} ({(~df['isActive']).sum()/len(df)*100:.1f}%)")
print(f"Average Age: {df['age'].mean():.1f} years")
print(f"Average Score: {df['score'].mean():.2f}")
print(f"Average Login Count: {df['loginCount'].mean():.0f}")
print(f"Date Range: {df['createdAt'].min().date()} to {df['createdAt'].max().date()}")
