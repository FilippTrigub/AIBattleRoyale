#!/usr/bin/env python3
"""
MongoDB Data Visualization Script (Optimized)
Uses aggregation pipelines to avoid loading all data into memory
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pymongo import MongoClient
import os
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

print("Connecting to MongoDB...")
# Get connection string from environment
connection_string = os.environ.get('MDB_MCP_CONNECTION_STRING')
client = MongoClient(connection_string)
db = client['testdb']
collection = db['dummy_data']

print(f"Connected to database. Total documents: {collection.count_documents({})}")

print("\nGenerating visualizations using aggregation pipelines...")

# 1. Age Distribution
print("Generating age distribution...")
age_pipeline = [
    {"$group": {"_id": "$age", "count": {"$sum": 1}}},
    {"$sort": {"_id": 1}}
]
age_data = list(collection.aggregate(age_pipeline))
ages = [d['_id'] for d in age_data]
age_counts = [d['count'] for d in age_data]

plt.figure(figsize=(12, 8))
plt.bar(ages, age_counts, color='skyblue', edgecolor='black', alpha=0.7, width=0.8)
plt.title('Age Distribution of Users', fontsize=16, fontweight='bold')
plt.xlabel('Age', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('/vercel/sandbox/age_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved: age_distribution.png")
plt.close()

# 2. Score Distribution (binned)
print("Generating score distribution...")
score_pipeline = [
    {"$bucket": {
        "groupBy": "$score",
        "boundaries": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "default": "Other",
        "output": {"count": {"$sum": 1}}
    }}
]
score_data = list(collection.aggregate(score_pipeline))
score_bins = [f"{d['_id']}-{d['_id']+10}" for d in score_data]
score_counts = [d['count'] for d in score_data]

plt.figure(figsize=(12, 8))
plt.bar(score_bins, score_counts, color='coral', edgecolor='black', alpha=0.7)
plt.title('Score Distribution', fontsize=16, fontweight='bold')
plt.xlabel('Score Range', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('/vercel/sandbox/score_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved: score_distribution.png")
plt.close()

# 3. Active vs Inactive Users
print("Generating activity status...")
activity_pipeline = [
    {"$group": {"_id": "$isActive", "count": {"$sum": 1}}}
]
activity_data = list(collection.aggregate(activity_pipeline))
labels = ['Inactive' if d['_id'] == False else 'Active' for d in activity_data]
counts = [d['count'] for d in activity_data]

plt.figure(figsize=(10, 8))
colors = ['#e74c3c', '#2ecc71']
plt.pie(counts, labels=labels, autopct='%1.1f%%', colors=colors, 
        startangle=90, textprops={'fontsize': 14})
plt.title('User Activity Status', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('/vercel/sandbox/activity_status.png', dpi=300, bbox_inches='tight')
print("✓ Saved: activity_status.png")
plt.close()

# 4. Theme Preferences
print("Generating theme preferences...")
theme_pipeline = [
    {"$group": {"_id": "$metadata.preferences.theme", "count": {"$sum": 1}}}
]
theme_data = list(collection.aggregate(theme_pipeline))
theme_labels = [d['_id'] for d in theme_data]
theme_counts = [d['count'] for d in theme_data]

plt.figure(figsize=(10, 8))
colors_theme = ['#34495e' if t == 'dark' else '#ecf0f1' for t in theme_labels]
plt.pie(theme_counts, labels=theme_labels, autopct='%1.1f%%', colors=colors_theme, 
        startangle=90, textprops={'fontsize': 14})
plt.title('Theme Preferences (Dark vs Light)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('/vercel/sandbox/theme_preferences.png', dpi=300, bbox_inches='tight')
print("✓ Saved: theme_preferences.png")
plt.close()

# 5. Language Distribution
print("Generating language distribution...")
language_pipeline = [
    {"$group": {"_id": "$metadata.preferences.language", "count": {"$sum": 1}}},
    {"$sort": {"count": -1}}
]
language_data = list(collection.aggregate(language_pipeline))
languages = [d['_id'] for d in language_data]
lang_counts = [d['count'] for d in language_data]

plt.figure(figsize=(12, 8))
plt.bar(languages, lang_counts, color='teal', edgecolor='black', alpha=0.7)
plt.title('Language Distribution', fontsize=16, fontweight='bold')
plt.xlabel('Language', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('/vercel/sandbox/language_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved: language_distribution.png")
plt.close()

# 6. Login Count Distribution (binned)
print("Generating login count distribution...")
login_pipeline = [
    {"$bucket": {
        "groupBy": "$metadata.loginCount",
        "boundaries": [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
        "default": "1000+",
        "output": {"count": {"$sum": 1}}
    }}
]
login_data = list(collection.aggregate(login_pipeline))
login_bins = [f"{d['_id']}-{d['_id']+100}" if isinstance(d['_id'], int) else d['_id'] for d in login_data]
login_counts = [d['count'] for d in login_data]

plt.figure(figsize=(12, 8))
plt.bar(login_bins, login_counts, color='purple', edgecolor='black', alpha=0.7)
plt.title('Login Count Distribution', fontsize=16, fontweight='bold')
plt.xlabel('Login Count Range', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('/vercel/sandbox/login_count_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved: login_count_distribution.png")
plt.close()

# 7. Notification Preferences
print("Generating notification preferences...")
notif_pipeline = [
    {"$group": {"_id": "$metadata.preferences.notifications", "count": {"$sum": 1}}}
]
notif_data = list(collection.aggregate(notif_pipeline))
notif_labels = ['Enabled' if d['_id'] == True else 'Disabled' for d in notif_data]
notif_counts = [d['count'] for d in notif_data]

plt.figure(figsize=(10, 8))
colors_notif = ['#3498db', '#95a5a6']
plt.pie(notif_counts, labels=notif_labels, autopct='%1.1f%%', colors=colors_notif, 
        startangle=90, textprops={'fontsize': 14})
plt.title('Notification Preferences', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('/vercel/sandbox/notification_preferences.png', dpi=300, bbox_inches='tight')
print("✓ Saved: notification_preferences.png")
plt.close()

# 8. Top 20 States by User Count
print("Generating top states...")
state_pipeline = [
    {"$group": {"_id": "$address.state", "count": {"$sum": 1}}},
    {"$sort": {"count": -1}},
    {"$limit": 20}
]
state_data = list(collection.aggregate(state_pipeline))
states = [d['_id'] for d in state_data]
state_counts = [d['count'] for d in state_data]

plt.figure(figsize=(14, 8))
plt.barh(range(len(states)), state_counts, color='steelblue', edgecolor='black', alpha=0.7)
plt.yticks(range(len(states)), states)
plt.title('Top 20 States by User Count', fontsize=16, fontweight='bold')
plt.xlabel('Number of Users', fontsize=12)
plt.ylabel('State', fontsize=12)
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('/vercel/sandbox/top_states.png', dpi=300, bbox_inches='tight')
print("✓ Saved: top_states.png")
plt.close()

# 9. Tag Count Distribution
print("Generating tag count distribution...")
tag_pipeline = [
    {"$project": {"tag_count": {"$size": "$tags"}}},
    {"$group": {"_id": "$tag_count", "count": {"$sum": 1}}},
    {"$sort": {"_id": 1}}
]
tag_data = list(collection.aggregate(tag_pipeline))
tag_counts_x = [d['_id'] for d in tag_data]
tag_counts_y = [d['count'] for d in tag_data]

plt.figure(figsize=(12, 8))
plt.bar(tag_counts_x, tag_counts_y, color='orange', edgecolor='black', alpha=0.7)
plt.title('Distribution of Tag Counts per User', fontsize=16, fontweight='bold')
plt.xlabel('Number of Tags', fontsize=12)
plt.ylabel('Number of Users', fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('/vercel/sandbox/tag_count_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved: tag_count_distribution.png")
plt.close()

# 10. Average Score by Activity Status
print("Generating score by activity...")
score_activity_pipeline = [
    {"$group": {
        "_id": "$isActive",
        "avg_score": {"$avg": "$score"},
        "min_score": {"$min": "$score"},
        "max_score": {"$max": "$score"}
    }}
]
score_activity_data = list(collection.aggregate(score_activity_pipeline))
activity_labels = ['Inactive' if d['_id'] == False else 'Active' for d in score_activity_data]
avg_scores = [d['avg_score'] for d in score_activity_data]

plt.figure(figsize=(10, 8))
plt.bar(activity_labels, avg_scores, color=['#e74c3c', '#2ecc71'], edgecolor='black', alpha=0.7)
plt.title('Average Score by Activity Status', fontsize=16, fontweight='bold')
plt.xlabel('Activity Status', fontsize=12)
plt.ylabel('Average Score', fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('/vercel/sandbox/score_by_activity.png', dpi=300, bbox_inches='tight')
print("✓ Saved: score_by_activity.png")
plt.close()

# 11. Average Age by Activity Status
print("Generating age by activity...")
age_activity_pipeline = [
    {"$group": {
        "_id": "$isActive",
        "avg_age": {"$avg": "$age"}
    }}
]
age_activity_data = list(collection.aggregate(age_activity_pipeline))
activity_labels_age = ['Inactive' if d['_id'] == False else 'Active' for d in age_activity_data]
avg_ages = [d['avg_age'] for d in age_activity_data]

plt.figure(figsize=(10, 8))
plt.bar(activity_labels_age, avg_ages, color=['#e74c3c', '#2ecc71'], edgecolor='black', alpha=0.7)
plt.title('Average Age by Activity Status', fontsize=16, fontweight='bold')
plt.xlabel('Activity Status', fontsize=12)
plt.ylabel('Average Age', fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('/vercel/sandbox/age_by_activity.png', dpi=300, bbox_inches='tight')
print("✓ Saved: age_by_activity.png")
plt.close()

# 12. Average Login Count by Theme
print("Generating login by theme...")
login_theme_pipeline = [
    {"$group": {
        "_id": "$metadata.preferences.theme",
        "avg_login": {"$avg": "$metadata.loginCount"}
    }}
]
login_theme_data = list(collection.aggregate(login_theme_pipeline))
themes = [d['_id'] for d in login_theme_data]
avg_logins = [d['avg_login'] for d in login_theme_data]

plt.figure(figsize=(10, 8))
colors_bar = ['#34495e' if t == 'dark' else '#ecf0f1' for t in themes]
plt.bar(themes, avg_logins, color=colors_bar, edgecolor='black', alpha=0.7)
plt.title('Average Login Count by Theme Preference', fontsize=16, fontweight='bold')
plt.xlabel('Theme', fontsize=12)
plt.ylabel('Average Login Count', fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('/vercel/sandbox/login_by_theme.png', dpi=300, bbox_inches='tight')
print("✓ Saved: login_by_theme.png")
plt.close()

# 13. User Registrations Over Time (by year)
print("Generating registrations over time...")
reg_pipeline = [
    {"$project": {
        "year": {"$year": "$createdAt"}
    }},
    {"$group": {"_id": "$year", "count": {"$sum": 1}}},
    {"$sort": {"_id": 1}}
]
reg_data = list(collection.aggregate(reg_pipeline))
years = [d['_id'] for d in reg_data]
reg_counts = [d['count'] for d in reg_data]

plt.figure(figsize=(14, 8))
plt.plot(years, reg_counts, linewidth=3, color='dodgerblue', marker='o', markersize=8)
plt.title('User Registrations Over Time (by Year)', fontsize=16, fontweight='bold')
plt.xlabel('Year', fontsize=12)
plt.ylabel('Number of Registrations', fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('/vercel/sandbox/registrations_over_time.png', dpi=300, bbox_inches='tight')
print("✓ Saved: registrations_over_time.png")
plt.close()

# 14. Age Groups vs Language Heatmap
print("Generating age-language heatmap...")
age_lang_pipeline = [
    {"$project": {
        "age_group": {
            "$switch": {
                "branches": [
                    {"case": {"$lte": ["$age", 25]}, "then": "18-25"},
                    {"case": {"$lte": ["$age", 35]}, "then": "26-35"},
                    {"case": {"$lte": ["$age", 45]}, "then": "36-45"},
                    {"case": {"$lte": ["$age", 55]}, "then": "46-55"},
                    {"case": {"$lte": ["$age", 65]}, "then": "56-65"}
                ],
                "default": "66+"
            }
        },
        "language": "$metadata.preferences.language"
    }},
    {"$group": {
        "_id": {"age_group": "$age_group", "language": "$language"},
        "count": {"$sum": 1}
    }}
]
age_lang_data = list(collection.aggregate(age_lang_pipeline))

# Create pivot table
age_groups = ['18-25', '26-35', '36-45', '46-55', '56-65', '66+']
languages_unique = sorted(set([d['_id']['language'] for d in age_lang_data]))
heatmap_matrix = np.zeros((len(age_groups), len(languages_unique)))

for d in age_lang_data:
    age_idx = age_groups.index(d['_id']['age_group'])
    lang_idx = languages_unique.index(d['_id']['language'])
    heatmap_matrix[age_idx][lang_idx] = d['count']

plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_matrix, annot=True, fmt='.0f', cmap='YlOrRd', 
            xticklabels=languages_unique, yticklabels=age_groups,
            cbar_kws={'label': 'Count'})
plt.title('User Count: Age Groups vs Language', fontsize=16, fontweight='bold')
plt.xlabel('Language', fontsize=12)
plt.ylabel('Age Group', fontsize=12)
plt.tight_layout()
plt.savefig('/vercel/sandbox/age_language_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Saved: age_language_heatmap.png")
plt.close()

# 15. Multi-panel Dashboard
print("Generating dashboard overview...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('MongoDB Data Dashboard Overview', fontsize=18, fontweight='bold')

# Panel 1: Age Distribution
axes[0, 0].bar(ages[:50], age_counts[:50], color='skyblue', edgecolor='black', alpha=0.7)
axes[0, 0].set_title('Age Distribution', fontweight='bold')
axes[0, 0].set_xlabel('Age')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].grid(axis='y', alpha=0.3)

# Panel 2: Activity Status
axes[0, 1].pie(counts, labels=labels, autopct='%1.1f%%', colors=['#e74c3c', '#2ecc71'], 
               startangle=90)
axes[0, 1].set_title('Activity Status', fontweight='bold')

# Panel 3: Language Distribution
axes[1, 0].bar(languages, lang_counts, color='teal', edgecolor='black', alpha=0.7)
axes[1, 0].set_title('Language Distribution', fontweight='bold')
axes[1, 0].set_xlabel('Language')
axes[1, 0].set_ylabel('Count')
axes[1, 0].grid(axis='y', alpha=0.3)

# Panel 4: Theme Preferences
axes[1, 1].bar(themes, avg_logins, color=colors_bar, edgecolor='black', alpha=0.7)
axes[1, 1].set_title('Avg Login by Theme', fontweight='bold')
axes[1, 1].set_xlabel('Theme')
axes[1, 1].set_ylabel('Avg Logins')
axes[1, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('/vercel/sandbox/dashboard_overview.png', dpi=300, bbox_inches='tight')
print("✓ Saved: dashboard_overview.png")
plt.close()

# Get summary statistics
print("\n" + "="*60)
print("Generating summary statistics...")
total_users = collection.count_documents({})
active_users = collection.count_documents({"isActive": True})
inactive_users = collection.count_documents({"isActive": False})

avg_stats = list(collection.aggregate([
    {"$group": {
        "_id": None,
        "avg_age": {"$avg": "$age"},
        "avg_score": {"$avg": "$score"},
        "avg_login": {"$avg": "$metadata.loginCount"}
    }}
]))[0]

date_range = list(collection.aggregate([
    {"$group": {
        "_id": None,
        "min_date": {"$min": "$createdAt"},
        "max_date": {"$max": "$createdAt"}
    }}
]))[0]

print("="*60)
print("All visualizations generated successfully!")
print("="*60)
print("\nGenerated PNG files:")
print("  1. age_distribution.png")
print("  2. score_distribution.png")
print("  3. activity_status.png")
print("  4. theme_preferences.png")
print("  5. language_distribution.png")
print("  6. login_count_distribution.png")
print("  7. notification_preferences.png")
print("  8. top_states.png")
print("  9. tag_count_distribution.png")
print(" 10. score_by_activity.png")
print(" 11. age_by_activity.png")
print(" 12. login_by_theme.png")
print(" 13. registrations_over_time.png")
print(" 14. age_language_heatmap.png")
print(" 15. dashboard_overview.png")
print("="*60)

print("\nData Summary Statistics:")
print(f"Total Users: {total_users:,}")
print(f"Active Users: {active_users:,} ({active_users/total_users*100:.1f}%)")
print(f"Inactive Users: {inactive_users:,} ({inactive_users/total_users*100:.1f}%)")
print(f"Average Age: {avg_stats['avg_age']:.1f} years")
print(f"Average Score: {avg_stats['avg_score']:.2f}")
print(f"Average Login Count: {avg_stats['avg_login']:.0f}")
print(f"Date Range: {date_range['min_date'].date()} to {date_range['max_date'].date()}")
print("="*60)

client.close()
