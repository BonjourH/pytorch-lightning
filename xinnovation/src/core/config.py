import os
import yaml
import json
import importlib.util
from typing import Dict, Any, Optional, List, Union, Iterator, Tuple
from omegaconf import OmegaConf, DictConfig

class Config:
    def __init__(
        self, 
        cfg_dict: Optional[Dict] = None, 
        filename: Optional[str] = None
    ):
        """
        初始化配置
        
        Args:
            cfg_dict (Dict, optional): 直接传入配置字典
            filename (str, optional): 配置文件路径
        """
        self._cfg_dict = {}
        
        if cfg_dict is not None:
            self._init_from_dict(cfg_dict)
        elif filename is not None:
            self._init_from_dict(self.from_file(filename))
        
        # 将配置转换为OmegaConf对象以获得更好的访问体验
        self._cfg_omega = OmegaConf.create(self._cfg_dict)

    def _init_from_dict(self, cfg_dict: Dict):
        """
        从字典初始化，支持嵌套配置
        
        Args:
            cfg_dict (Dict): 配置字典
        """
        for key, value in cfg_dict.items():
            if isinstance(value, dict):
                # 对于嵌套字典，创建子 Config 实例
                setattr(self, key, Config(value))
                self._cfg_dict[key] = getattr(self, key)._cfg_dict  # 存储子配置的字典表示
            else:
                # 对于普通值，直接设置
                setattr(self, key, value)
                self._cfg_dict[key] = value

    @staticmethod
    def from_file(filename: str) -> 'Config':
        """
        从单个文件创建配置字典的静态方法
        
        Args:
            filename (str): 配置文件路径
        
        Returns:
            Dict: 解析后的配置字典
        """
        file_ext = os.path.splitext(filename)[1].lower()
        cfg_dict = {}
        
        with open(filename, 'r', encoding='utf-8') as f:
            if file_ext in ['.yaml', '.yml']:
                cfg_dict = yaml.safe_load(f) or {}
            elif file_ext == '.json':
                cfg_dict = json.load(f)
            elif file_ext == '.py':
                cfg_dict = Config._parse_py_file(filename)
            else:
                raise ValueError(f'Unsupported config file type: {file_ext}')
        return Config(cfg_dict)

    @classmethod
    def from_files(cls, filenames: Union[List[str], str], merge: bool = True) -> 'Config':
        """
        从多个文件创建配置实例的类方法
        
        Args:
            filenames (Union[List[str], str]): 配置文件路径列表或单个路径
            merge (bool, optional): 是否合并多个文件的配置。默认为 True
        
        Returns:
            Config: 配置实例
        """
        # 处理单个文件路径的情况
        if isinstance(filenames, str):
            filenames = [filenames]
        
        # 解析所有文件
        configs = [cls.from_file(filename) for filename in filenames]
        
        # 如果不需要合并，抛出异常
        if not merge:
            raise ValueError("Use merge=True when using from_files")
        
        # 合并配置字典
        merged_config = {}
        for cfg in configs:
            merged_config.update(cfg)
        
        # 返回 Config 实例
        return cls(merged_config)

    @staticmethod
    def _parse_py_file(filename: str) -> Dict:
        """解析Python配置文件"""
        module_name = os.path.splitext(os.path.basename(filename))[0]
        spec = importlib.util.spec_from_file_location(module_name, filename)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Filter out types from typing module, built-in symbols, and unsupported types
        result = {}
        for k, v in module.__dict__.items():
            # Skip built-ins, private attrs, imports from typing, and callable objects
            if (k.startswith('__') or 
                k in ('List', 'Dict', 'Optional', 'Union', 'Any') or
                callable(v) or
                getattr(v, '__module__', '').startswith('typing')):
                continue
            result[k] = v
            
        return result

    def __getitem__(self, key: str) -> Any:
        """像字典一样通过 [] 获取值"""
        return self._cfg_dict[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """像字典一样通过 [] 设置值"""
        setattr(self, key, value)
        self._cfg_dict[key] = value
        # 更新OmegaConf对象
        self._cfg_omega = OmegaConf.create(self._cfg_dict)

    def keys(self):
        """返回所有键"""
        return self._cfg_dict.keys()

    def values(self):
        """返回所有值"""
        return self._cfg_dict.values()

    def items(self):
        """返回所有键值对"""
        return self._cfg_dict.items()

    def get(self, key: str, default: Any = None) -> Any:
        """获取值，带默认值"""
        return self._cfg_dict.get(key, default)

    def update(self, cfg_dict: Dict) -> None:
        """更新配置"""
        self._init_from_dict(cfg_dict)
        # 更新OmegaConf对象
        self._cfg_omega = OmegaConf.create(self._cfg_dict)

    def to_dict(self) -> Dict:
        """将配置转换为普通字典"""
        return self._cfg_dict

    def to_omega(self) -> DictConfig:
        """获取OmegaConf表示"""
        return self._cfg_omega
    
    def copy(self) -> 'Config':
        """复制配置"""
        return Config(self._cfg_dict)
    
    def __deepcopy__(self, memo: Dict) -> 'Config':
        """深度复制配置"""
        return Config(self._cfg_dict)
    
    def pop(self, key: str) -> Any:
        """删除并返回指定键的值"""
        return self._cfg_dict.pop(key)
    
    def popitem(self) -> Tuple[str, Any]:
        """删除并返回最后一对键值对"""
        return self._cfg_dict.popitem()

    def save_to_file(self, filename: str) -> None:
        """
        保存配置到文件
        
        Args:
            filename (str): 要保存的文件路径
        """
        file_ext = os.path.splitext(filename)[1].lower()
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        
        with open(filename, 'w', encoding='utf-8') as f:
            if file_ext in ['.yaml', '.yml']:
                yaml.dump(self._cfg_dict, f, default_flow_style=False, sort_keys=False)
            elif file_ext == '.json':
                json.dump(self._cfg_dict, f, indent=2)
            else:
                raise ValueError(f'Unsupported config file type for saving: {file_ext}')

    def __repr__(self) -> str:
        """字符串表示"""
        return repr(self._cfg_dict)
    
    def __str__(self) -> str:
        """美化后的字符串表示"""
        return yaml.dump(self._cfg_dict, default_flow_style=False, sort_keys=False)
    
    @property
    def config(self) -> DictConfig:
        """获取OmegaConf配置对象，用于与需要该类型的函数集成"""
        return self._cfg_omega
 