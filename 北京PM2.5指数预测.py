import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1.数据加载与检查
def load_and_inspect_data():
    print("="*60)
    print("数据加载与检查")
    print("="*60)

    train_df = pd.read_csv(r"train.csv")
    test_df = pd.read_csv(r"test_noLabel.csv")

    print(f"训练数据形状: {train_df.shape}")
    print(f"测试数据形状: {test_df.shape}")
    print("\n训练集统计信息:")
    print(train_df.describe())

    # 检查缺失值
    print("\n训练集缺失值检查:")
    missing_cols = train_df.columns[train_df.isnull().sum() > 0]
    if len(missing_cols) > 0:
        print(train_df[missing_cols].isnull().sum())
    else:
        print("无缺失值")

    # 检查重复值
    print(f"\n训练集重复行数: {train_df.duplicated().sum()}")

    return train_df, test_df

# 2. 数据探索性分析
def exploratory_data_analysis(train_df):
    print("\n" + "="*60)
    print("数据探索分析")
    print("="*60)

    fig, axes = plt.subplots(3, 3, figsize=(20, 16))

    # PM2.5分布
    axes[0,0].hist(train_df['Label'].dropna(), bins=60, color='skyblue', alpha=0.7, edgecolor='k', density=True)
    train_df['Label'].plot(kind='kde', ax=axes[0,0], secondary_y=True, color='red', alpha=0.7)
    axes[0,0].set_title('PM2.5浓度分布（直方图+KDE）')
    axes[0,0].axvline(train_df['Label'].mean(), color='red', linestyle='--', label='均值')
    axes[0,0].axvline(train_df['Label'].median(), color='green', linestyle=':', label='中位数')
    axes[0,0].legend()

    # 箱线图
    axes[0,1].boxplot(train_df['Label'].dropna(), vert=False, patch_artist=True)
    axes[0,1].set_title('PM2.5箱线图')
    axes[0,1].set_xlabel('PM2.5浓度')

    # 时间序列趋势
    if 'date' in train_df.columns and 'hour' in train_df.columns:
        train_df['datetime'] = pd.to_datetime(train_df['date']) + pd.to_timedelta(train_df['hour'], unit='h')
        if len(train_df) > 1000:
            sample_idx = np.random.choice(len(train_df), min(1000, len(train_df)), replace=False)
            time_sample = train_df.iloc[sample_idx].sort_values('datetime')
        else:
            time_sample = train_df.sort_values('datetime')

        axes[0,2].scatter(time_sample['datetime'], time_sample['Label'], alpha=0.5, s=5)
        axes[0,2].set_title('PM2.5时间序列趋势')
        axes[0,2].set_xlabel('时间')
        axes[0,2].set_ylabel('PM2.5')
        axes[0,2].tick_params(axis='x', rotation=45)
    else:
        axes[0,2].text(0.5, 0.5, '无时间列', ha='center', va='center', transform=axes[0,2].transAxes)
        axes[0,2].set_title('PM2.5时间序列趋势')

    # 温度 vs PM2.5
    if 'TEMP' in train_df.columns:
        axes[1,0].scatter(train_df['TEMP'], train_df['Label'], alpha=0.4, s=10, c='green')

        # 添加回归线
        valid_idx = train_df['TEMP'].notna() & train_df['Label'].notna()
        if valid_idx.sum() > 0:
            z = np.polyfit(train_df.loc[valid_idx, 'TEMP'], train_df.loc[valid_idx, 'Label'], 1)
            p = np.poly1d(z)
            x_sorted = np.sort(train_df.loc[valid_idx, 'TEMP'])
            axes[1,0].plot(x_sorted, p(x_sorted), "r--", alpha=0.8, linewidth=2)

        axes[1,0].set_xlabel('温度')
        axes[1,0].set_ylabel('PM2.5')
        axes[1,0].set_title('温度 vs PM2.5（带回归线）')
    else:
        axes[1,0].text(0.5, 0.5, '无温度数据', ha='center', va='center', transform=axes[1,0].transAxes)
        axes[1,0].set_title('温度 vs PM2.5')

    # 累积风速 vs PM2.5
    if 'Iws' in train_df.columns:
        axes[1,1].scatter(train_df['Iws'], train_df['Label'], alpha=0.4, s=10, c='purple')
        axes[1,1].set_xlabel('Iws')
        axes[1,1].set_ylabel('PM2.5')
        axes[1,1].set_title('累积风速 vs PM2.5')
    else:
        axes[1,1].text(0.5, 0.5, '无风速数据', ha='center', va='center', transform=axes[1,1].transAxes)
        axes[1,1].set_title('累积风速 vs PM2.5')

    # 露点温度 vs PM2.5
    if 'DEWP' in train_df.columns:
        axes[1,2].scatter(train_df['DEWP'], train_df['Label'], alpha=0.4, s=10, c='orange')
        axes[1,2].set_xlabel('DEWP（露点）')
        axes[1,2].set_ylabel('PM2.5')
        axes[1,2].set_title('露点温度 vs PM2.5')
    else:
        axes[1,2].text(0.5, 0.5, '无露点数据', ha='center', va='center', transform=axes[1,2].transAxes)
        axes[1,2].set_title('露点温度 vs PM2.5')

    # 小时趋势
    if 'hour' in train_df.columns:
        hourly_stats = train_df.groupby('hour')['Label'].agg(['mean', 'std', 'count'])
        axes[2,0].plot(hourly_stats.index, hourly_stats['mean'], marker='o', color='orange', label='均值')
        axes[2,0].fill_between(hourly_stats.index,
                               hourly_stats['mean'] - hourly_stats['std'],
                               hourly_stats['mean'] + hourly_stats['std'],
                               alpha=0.2, color='orange')
        axes[2,0].set_title('小时平均PM2.5（±标准差）')
        axes[2,0].set_xlabel('小时')
        axes[2,0].set_ylabel('PM2.5')
        axes[2,0].grid(True, alpha=0.3)
        axes[2,0].legend()
    else:
        axes[2,0].text(0.5, 0.5, '无小时数据', ha='center', va='center', transform=axes[2,0].transAxes)
        axes[2,0].set_title('小时平均PM2.5')

    # 月份趋势
    if 'date' in train_df.columns:
        train_df['month'] = pd.to_datetime(train_df['date']).dt.month
        monthly_avg = train_df.groupby('month')['Label'].mean()
        axes[2,1].plot(monthly_avg.index, monthly_avg.values, marker='s', color='brown', linewidth=2)
        axes[2,1].set_title('月平均PM2.5')
        axes[2,1].set_xlabel('月份')
        axes[2,1].set_ylabel('PM2.5')
        axes[2,1].grid(True, alpha=0.3)
    else:
        axes[2,1].text(0.5, 0.5, '无日期数据', ha='center', va='center', transform=axes[2,1].transAxes)
        axes[2,1].set_title('月平均PM2.5')

    # 相关性热力图
    corr_cols = ['Label', 'DEWP', 'TEMP', 'PRES', 'Iws', 'Ir', 'Is']
    available_cols = [col for col in corr_cols if col in train_df.columns]
    if len(available_cols) > 1:
        corr_matrix = train_df[available_cols].corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                    ax=axes[2,2], mask=mask, fmt='.2f', square=True)
        axes[2,2].set_title('特征相关性热力图')
    else:
        axes[2,2].text(0.5, 0.5, '特征不足\n无法计算相关性', ha='center', va='center', transform=axes[2,2].transAxes)
        axes[2,2].set_title('特征相关性热力图')

    plt.tight_layout()
    plt.savefig('enhanced_eda.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 异常值检测
    if 'Label' in train_df.columns:
        print("\n异常值检测（基于IQR）:")
        Q1 = train_df['Label'].quantile(0.25)
        Q3 = train_df['Label'].quantile(0.75)
        IQR = Q3 - Q1
        outliers = train_df[(train_df['Label'] < Q1 - 1.5*IQR) | (train_df['Label'] > Q3 + 1.5*IQR)]
        print(f"异常值数量: {len(outliers)} ({len(outliers)/len(train_df)*100:.2f}%)")

        print(f"\nPM2.5 详细统计信息:")
        stats = train_df['Label'].describe()
        stats['skew'] = train_df['Label'].skew()
        stats['kurtosis'] = train_df['Label'].kurtosis()
        print(stats)
    else:
        print("\n警告: 数据集中没有Label列")

    return train_df

# 3. 特征工程
def advanced_feature_engineering_enhanced(df, is_train=True, target_col='Label'):
    df_proc = df.copy()

    # 记录原始特征
    original_features = df_proc.columns.tolist()

    # 处理时间特征
    if 'date' in df_proc.columns:
        df_proc['date'] = pd.to_datetime(df_proc['date'], errors='coerce')

        #  时间特征增强
        df_proc['year'] = df_proc['date'].dt.year
        df_proc['month'] = df_proc['date'].dt.month
        df_proc['day'] = df_proc['date'].dt.day
        df_proc['dayofweek'] = df_proc['date'].dt.dayofweek
        df_proc['dayofyear'] = df_proc['date'].dt.dayofyear
        df_proc['weekofyear'] = df_proc['date'].dt.isocalendar().week
        df_proc['quarter'] = df_proc['date'].dt.quarter
        df_proc['is_weekend'] = (df_proc['dayofweek'] >= 5).astype(int)
        df_proc['is_month_start'] = df_proc['date'].dt.is_month_start.astype(int)
        df_proc['is_month_end'] = df_proc['date'].dt.is_month_end.astype(int)

        # 季节编码
        df_proc['season'] = ((df_proc['month'] % 12 + 3) // 3).astype(int)

        # 周期性编码
        if 'hour' in df_proc.columns:
            df_proc['hour_sin'] = np.sin(2 * np.pi * df_proc['hour'] / 24)
            df_proc['hour_cos'] = np.cos(2 * np.pi * df_proc['hour'] / 24)
        if 'month' in df_proc.columns:
            df_proc['month_sin'] = np.sin(2 * np.pi * df_proc['month'] / 12)
            df_proc['month_cos'] = np.cos(2 * np.pi * df_proc['month'] / 12)
        df_proc['dayofweek_sin'] = np.sin(2 * np.pi * df_proc['dayofweek'] / 7)
        df_proc['dayofweek_cos'] = np.cos(2 * np.pi * df_proc['dayofweek'] / 7)
        df_proc['dayofyear_sin'] = np.sin(2 * np.pi * df_proc['dayofyear'] / 365)
        df_proc['dayofyear_cos'] = np.cos(2 * np.pi * df_proc['dayofyear'] / 365)

    # 气象特征增强
    if all(col in df_proc.columns for col in ['TEMP', 'DEWP']):
        # 温度相关
        df_proc['temp_dewp_diff'] = df_proc['TEMP'] - df_proc['DEWP']
        df_proc['temp_dewp_ratio'] = df_proc['DEWP'] / (df_proc['TEMP'] + 1e-6)
        df_proc['apparent_temp'] = df_proc['TEMP'] + 0.33 * df_proc['DEWP']  # 体感温度近似

        # 热指数
        df_proc['heat_index'] = 0.5 * (df_proc['TEMP'] + 61.0 + (df_proc['TEMP'] - 68.0) * 1.2 + df_proc['DEWP'] * 0.094)

    if 'PRES' in df_proc.columns:
        # 气压标准化
        df_proc['pres_normalized'] = (df_proc['PRES'] - df_proc['PRES'].mean()) / (df_proc['PRES'].std() + 1e-6)

    if 'Iws' in df_proc.columns:
        # 风速特征
        df_proc['log_iws'] = np.log1p(df_proc['Iws'])
        df_proc['sqrt_iws'] = np.sqrt(df_proc['Iws'].clip(lower=0))

        # 风速分箱
        try:
            df_proc['wind_speed_bin'] = pd.cut(df_proc['Iws'],
                                              bins=[-np.inf, 1, 5, 10, 20, np.inf],
                                              labels=[0, 1, 2, 3, 4])
        except:
            # 如果分箱失败，使用等频分箱
            df_proc['wind_speed_bin'] = pd.qcut(df_proc['Iws'], q=5, labels=False, duplicates='drop')

    # 风向特征
    wind_dir_cols = [col for col in df_proc.columns if 'cbwd' in col.lower()]
    if wind_dir_cols:
        df_proc['wind_dir_count'] = df_proc[wind_dir_cols].sum(axis=1)

    # 天气现象组合
    if all(col in df_proc.columns for col in ['Ir', 'Is']):
        df_proc['has_precipitation'] = ((df_proc['Ir'] > 0) | (df_proc['Is'] > 0)).astype(int)
        df_proc['precip_intensity'] = df_proc['Ir'] + df_proc['Is']
        df_proc['precip_type'] = np.where(df_proc['Is'] > df_proc['Ir'], 2,
                                         np.where(df_proc['Ir'] > 0, 1, 0))

    # 时间窗口统计特征
    if is_train and target_col in df_proc.columns and 'date' in df_proc.columns and 'hour' in df_proc.columns:
        # 对训练集创建时间序列特征
        df_proc['datetime'] = pd.to_datetime(df_proc['date']) + pd.to_timedelta(df_proc['hour'], unit='h')
        df_sorted = df_proc.sort_values('datetime').reset_index(drop=True)

        # 对数值型特征进行滚动统计
        numeric_cols = []
        for col in ['TEMP', 'DEWP', 'PRES', 'Iws', 'Ir', 'Is']:
            if col in df_sorted.columns:
                numeric_cols.append(col)

        for col in numeric_cols:
            for window in [3, 6, 12, 24]:  # 不同时间窗口
                df_sorted[f'{col}_roll_mean_{window}'] = df_sorted[col].rolling(window=window, min_periods=1).mean()
                df_sorted[f'{col}_roll_std_{window}'] = df_sorted[col].rolling(window=window, min_periods=1).std().fillna(0)

        # 差分特征
        for col in numeric_cols:
            for lag in [1, 3, 6]:
                df_sorted[f'{col}_diff_{lag}'] = df_sorted[col].diff(lag)

        df_proc = df_sorted

    # 交互特征
    if all(col in df_proc.columns for col in ['TEMP', 'DEWP', 'PRES']):
        # 气象指数
        df_proc['temp_pres_ratio'] = df_proc['TEMP'] / (df_proc['PRES'] + 1e-6)
        df_proc['dewp_pres_ratio'] = df_proc['DEWP'] / (df_proc['PRES'] + 1e-6)

        # 温湿交互
        if 'temp_dewp_diff' in df_proc.columns:
            df_proc['temp_humidity_interaction'] = df_proc['TEMP'] * df_proc['temp_dewp_diff']

        # 风温交互
        if 'Iws' in df_proc.columns:
            df_proc['wind_chill'] = 13.12 + 0.6215 * df_proc['TEMP'] - 11.37 * (df_proc['Iws']**0.16) + 0.3965 * df_proc['TEMP'] * (df_proc['Iws']**0.16)
            df_proc['temp_wind_interaction'] = df_proc['TEMP'] * df_proc['Iws']

    # 多项式特征
    poly_features = []
    for col in ['TEMP', 'DEWP', 'PRES', 'Iws']:
        if col in df_proc.columns:
            df_proc[f'{col}_squared'] = df_proc[col] ** 2
            df_proc[f'{col}_cubed'] = df_proc[col] ** 3
            df_proc[f'{col}_sqrt'] = np.sqrt(np.abs(df_proc[col]))
            poly_features.extend([f'{col}_squared', f'{col}_cubed', f'{col}_sqrt'])

    # 统计聚合特征
    if len(df_proc) > 100:  # 仅在数据量足够时
        for col in ['TEMP', 'DEWP', 'PRES', 'Iws']:
            if col in df_proc.columns:
                df_proc[f'{col}_zscore'] = (df_proc[col] - df_proc[col].mean()) / (df_proc[col].std() + 1e-6)

    return df_proc

# 4. 智能预处理与特征选择
def enhanced_preprocess_data(train_df, test_df):
    print("\n" + "="*60)
    print("智能数据预处理与特征工程")
    print("="*60)

    # 确保有datetime列
    if 'date' in train_df.columns and 'hour' in train_df.columns:
        train_df['datetime'] = pd.to_datetime(train_df['date']) + pd.to_timedelta(train_df['hour'], unit='h')
        test_df['datetime'] = pd.to_datetime(test_df['date']) + pd.to_timedelta(test_df['hour'], unit='h')
    else:
        print("警告: 数据中缺少date或hour列，无法创建datetime列")

    # 分别进行特征工程
    print("对训练集进行特征工程...")
    train_fe = advanced_feature_engineering_enhanced(train_df, is_train=True)

    print("对测试集进行特征工程...")
    test_fe = advanced_feature_engineering_enhanced(test_df, is_train=False)

    # 分离特征和目标
    if 'Label' in train_fe.columns:
        y_train = train_fe['Label'].values
    else:
        raise ValueError("训练数据中没有Label列！")

    # 删除不用的列
    drop_cols = ['ID', 'date', 'datetime', 'Label']
    drop_cols = [c for c in drop_cols if c in train_fe.columns]

    X_train_raw = train_fe.drop(columns=drop_cols, errors='ignore')
    X_test_raw = test_fe.drop(columns=[c for c in drop_cols if c in test_fe.columns], errors='ignore')

    # 对齐特征
    print(f"\n训练集原始特征数: {X_train_raw.shape[1]}")
    print(f"测试集原始特征数: {X_test_raw.shape[1]}")

    # 找出共同特征
    common_features = list(set(X_train_raw.columns) & set(X_test_raw.columns))
    print(f"共同特征数: {len(common_features)}")

    # 只使用共同特征
    X_train = X_train_raw[common_features].copy()
    X_test = X_test_raw[common_features].copy()

    # 处理缺失值
    print("\n处理缺失值...")
    for col in X_train.columns:
        if X_train[col].isnull().any() or (col in X_test.columns and X_test[col].isnull().any()):
            # 使用中位数填充数值特征
            if X_train[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                med_val = X_train[col].median()
                X_train[col].fillna(med_val, inplace=True)
                if col in X_test.columns:
                    X_test[col].fillna(med_val, inplace=True)
            else:
                # 对于非数值特征，使用众数
                mode_val = X_train[col].mode()[0] if not X_train[col].mode().empty else 0
                X_train[col].fillna(mode_val, inplace=True)
                if col in X_test.columns:
                    X_test[col].fillna(mode_val, inplace=True)

    # 异常值处理（基于IQR）- 仅对数值特征
    print("\n处理异常值...")
    for col in X_train.select_dtypes(include=[np.number]).columns:
        if col not in ['hour_sin', 'hour_cos', 'month_sin', 'month_cos',
                      'dayofweek_sin', 'dayofweek_cos', 'dayofyear_sin', 'dayofyear_cos']:
            Q1 = X_train[col].quantile(0.01)
            Q3 = X_train[col].quantile(0.99)
            IQR = Q3 - Q1

            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR

            # 对训练集和测试集进行缩尾处理
            X_train[col] = X_train[col].clip(lower_bound, upper_bound)
            if col in X_test.columns:
                X_test[col] = X_test[col].clip(lower_bound, upper_bound)

    # 特征缩放
    from sklearn.preprocessing import RobustScaler
    print("\n特征缩放...")
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

    # 跳过已标准化和周期性特征
    exclude_cols = ['hour_sin', 'hour_cos', 'month_sin', 'month_cos',
                   'dayofweek_sin', 'dayofweek_cos', 'dayofyear_sin', 'dayofyear_cos']
    scale_cols = [col for col in numeric_cols if col not in exclude_cols and not any(x in col for x in ['_zscore', '_normalized'])]

    if scale_cols:
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train[scale_cols])
        X_test_scaled = scaler.transform(X_test[scale_cols])

        # 将缩放后的值放回
        X_train[scale_cols] = X_train_scaled
        X_test[scale_cols] = X_test_scaled

    # 类型转换
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    feature_names = list(X_train.columns)
    print(f"\n最终特征维度: X_train={X_train.shape}, X_test={X_test.shape}, y_train={y_train.shape}")
    print(f"特征数量: {len(feature_names)}")

    return X_train.values, y_train, X_test.values, feature_names, test_df['ID'].values


# 5. 增强集成模型
def train_ensemble_models(X_train, y_train):
    from sklearn.ensemble import StackingRegressor, RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor
    from sklearn.model_selection import KFold, cross_val_score

    print("\n" + "="*60)
    print("训练集成模型")
    print("="*60)

    # 定义模型参数
    lgb_params = {
        'n_estimators': 1000,
        'max_depth': 8,
        'learning_rate': 0.05,
        'num_leaves': 63,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.3,
        'reg_lambda': 0.7,
        'min_child_samples': 20,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }

    xgb_params = {
        'n_estimators': 1000,
        'max_depth': 7,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'min_child_weight': 3,
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 0
    }

    rf_params = {
        'n_estimators': 300,
        'max_depth': 12,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'max_features': 'sqrt',
        'bootstrap': True,
        'random_state': 42,
        'n_jobs': -1
    }

    gb_params = {
        'n_estimators': 300,
        'learning_rate': 0.05,
        'max_depth': 6,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'max_features': 'sqrt',
        'random_state': 42
    }

    # 创建基础模型
    print("创建基础模型...")
    lgb_model = LGBMRegressor(**lgb_params)
    xgb_model = XGBRegressor(**xgb_params)
    rf_model = RandomForestRegressor(**rf_params)
    gb_model = GradientBoostingRegressor(**gb_params)

    # 第一层模型
    level0 = [
        ('lgb', lgb_model),
        ('xgb', xgb_model),
        ('rf', rf_model),
        ('gb', gb_model)
    ]

    # 第二层模型
    level1 = Ridge(alpha=1.0)

    # 创建Stacking模型
    stacking_model = StackingRegressor(
        estimators=level0,
        final_estimator=level1,
        cv=5,
        passthrough=False,
        n_jobs=-1
    )

    # 训练Stacking模型
    print("训练Stacking集成模型...")
    stacking_model.fit(X_train, y_train)

    # 单独训练每个基模型用于加权融合
    print("\n训练单独的基模型...")
    models_dict = {}

    models_dict['lgb'] = LGBMRegressor(**lgb_params)
    models_dict['xgb'] = XGBRegressor(**xgb_params)
    models_dict['rf'] = RandomForestRegressor(**rf_params)
    models_dict['gb'] = GradientBoostingRegressor(**gb_params)

    for name, model in models_dict.items():
        print(f"训练 {name}...")
        model.fit(X_train, y_train)

    # 交叉验证评估
    print("\n进行交叉验证评估...")
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    try:
        cv_scores = cross_val_score(
            stacking_model, X_train, y_train,
            cv=kfold,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1,
            verbose=0
        )

        rmse_scores = -cv_scores
        print(f"Stacking模型交叉验证RMSE: {rmse_scores.mean():.4f} (±{rmse_scores.std():.4f})")
    except Exception as e:
        print(f"交叉验证时出错: {e}")

    return stacking_model, models_dict


# 6. 加权集成预测
def weighted_ensemble_predict(models, X_test, test_ids, X_train, y_train):
    print("\n" + "="*60)
    print("加权集成预测")
    print("="*60)

    stacking_model, individual_models = models

    # 获取各个模型的预测
    print("获取各个模型的预测...")
    predictions = {}

    # Stacking模型预测
    print("Stacking模型预测...")
    pred_stacking = stacking_model.predict(X_test)
    predictions['stacking'] = pred_stacking

    # 单独模型预测
    for name, model in individual_models.items():
        print(f"{name}模型预测...")
        predictions[name] = model.predict(X_test)

    # 计算验证集性能来设置权重
    print("\n计算模型权重...")
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error

    # 划分验证集
    X_train_val, X_test_val, y_train_val, y_test_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, shuffle=True
    )

    val_scores = {}

    # 评估Stacking模型
    stacking_model_val = stacking_model
    stacking_val_pred = stacking_model_val.predict(X_test_val)
    stacking_rmse = np.sqrt(mean_squared_error(y_test_val, stacking_val_pred))
    val_scores['stacking'] = 1.0 / (stacking_rmse + 1e-8)
    print(f"Stacking RMSE: {stacking_rmse:.4f}")

    # 评估各个基模型
    for name, model in individual_models.items():
        # 重新训练模型
        model.fit(X_train_val, y_train_val)
        val_pred = model.predict(X_test_val)
        rmse = np.sqrt(mean_squared_error(y_test_val, val_pred))
        val_scores[name] = 1.0 / (rmse + 1e-8)
        print(f"{name} RMSE: {rmse:.4f}")

    # 归一化权重
    total_weight = sum(val_scores.values())
    weights = {k: v/total_weight for k, v in val_scores.items()}

    print("\n模型权重:")
    for name, weight in weights.items():
        print(f"{name}: {weight:.4f}")

    # 加权融合
    weighted_pred = np.zeros_like(pred_stacking)
    for name, pred in predictions.items():
        weighted_pred += pred * weights[name]

    # 后处理
    print("\n进行后处理...")

    # 分位数裁剪
    q99_5 = np.percentile(y_train, 99.5)
    q0_5 = np.percentile(y_train, 0.5)
    weighted_pred = np.clip(weighted_pred, q0_5, q99_5)

    # 时间序列平滑
    if len(weighted_pred) > 10:
        try:
            # 使用指数平滑
            ser = pd.Series(weighted_pred)
            smoothed = ser.ewm(span=3).mean().values
            weighted_pred = 0.8 * weighted_pred + 0.2 * smoothed
        except:
            pass

    # 残差修正
    train_pred = stacking_model.predict(X_train)
    residuals = y_train - train_pred
    mean_residual = residuals.mean()

    if abs(mean_residual) > np.std(residuals) * 0.1:
        print(f"应用残差修正: {mean_residual:.4f}")
        weighted_pred = weighted_pred - mean_residual

    # 确保正值
    weighted_pred = np.maximum(weighted_pred, 0)

    # 四舍五入
    weighted_pred = np.round(weighted_pred, 4)

    # 创建结果DataFrame
    results_df = pd.DataFrame({
        'ID': test_ids,
        'Label': weighted_pred
    })

    # 分析预测结果
    print(f"\n预测结果统计:")
    print(f"最小值: {weighted_pred.min():.2f}")
    print(f"最大值: {weighted_pred.max():.2f}")
    print(f"均值: {weighted_pred.mean():.2f}")
    print(f"标准差: {weighted_pred.std():.2f}")
    print(f"中位数: {np.median(weighted_pred):.2f}")

    print(f"\n与训练集分布对比:")
    print(f"训练集均值: {y_train.mean():.2f}, 预测均值: {weighted_pred.mean():.2f}")
    print(f"训练集标准差: {y_train.std():.2f}, 预测标准差: {weighted_pred.std():.2f}")

    return results_df

# 7. 特征重要性分析
def analyze_feature_importance(models, feature_names, X_train, y_train):
    print("\n" + "="*60)
    print("特征重要性分析")
    print("="*60)

    stacking_model, individual_models = models

    # LGBM特征重要性
    if 'lgb' in individual_models:
        lgb_model = individual_models['lgb']
        if hasattr(lgb_model, 'feature_importances_'):
            importances = lgb_model.feature_importances_
            indices = np.argsort(importances)[::-1]

            print("\nTop 20 特征重要性 (LGBM):")
            for i in range(min(20, len(feature_names))):
                print(f"{i+1:2d}. {feature_names[indices[i]]:30s} {importances[indices[i]]:.4f}")

            # 可视化
            plt.figure(figsize=(12, 8))
            top_n = min(20, len(feature_names))
            plt.barh(range(top_n), importances[indices[:top_n]])
            plt.yticks(range(top_n), [feature_names[i] for i in indices[:top_n]])
            plt.xlabel('特征重要性')
            plt.title('Top 20 特征重要性 (LGBM)')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig('feature_importance_lgb.png', dpi=300, bbox_inches='tight')
            plt.show()

    # 也可以检查XGBoost特征重要性
    if 'xgb' in individual_models:
        xgb_model = individual_models['xgb']
        if hasattr(xgb_model, 'feature_importances_'):
            xgb_importances = xgb_model.feature_importances_
            xgb_indices = np.argsort(xgb_importances)[::-1]

            print("\nTop 20 特征重要性 (XGBoost):")
            for i in range(min(20, len(feature_names))):
                print(f"{i+1:2d}. {feature_names[xgb_indices[i]]:30s} {xgb_importances[xgb_indices[i]]:.4f}")


# 8. 主函数
def main_enhanced():
    try:
        import time
        start_time = time.time()

        # 1. 加载数据
        train_df, test_df = load_and_inspect_data()

        # 2. 探索性分析
        print("\n" + "="*60)
        print("进行探索性数据分析...")
        print("="*60)
        train_df = exploratory_data_analysis(train_df)

        # 3. 特征工程与预处理
        X_train, y_train, X_test, feature_names, test_ids = enhanced_preprocess_data(train_df, test_df)

        # 4. 训练集成模型
        models = train_ensemble_models(X_train, y_train)

        # 5. 特征重要性分析
        if len(feature_names) > 0:
            analyze_feature_importance(models, feature_names, X_train, y_train)

        # 6. 加权集成预测
        predictions_df = weighted_ensemble_predict(models, X_test, test_ids, X_train, y_train)

        # 7. 保存结果
        output_file = 'pm25_predictions.csv'
        predictions_df.to_csv(output_file, index=False)

        # 8. 结果分析
        end_time = time.time()

        print("\n" + "="*60)
        print("预测完成！")
        print("="*60)
        print(f"预测结果已保存至: {output_file}")
        print(f"预测样本数: {len(predictions_df)}")
        print(f"总耗时: {end_time - start_time:.2f} 秒")

        # 显示预测分布
        plt.figure(figsize=(10, 6))
        plt.hist(predictions_df['Label'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(predictions_df['Label'].mean(), color='red', linestyle='--', label='预测均值')
        plt.axvline(np.median(predictions_df['Label']), color='green', linestyle=':', label='预测中位数')
        plt.xlabel('PM2.5预测值')
        plt.ylabel('频数')
        plt.title('预测值分布')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('predictions_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()

    except Exception as e:
        print(f"\n 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main_enhanced()