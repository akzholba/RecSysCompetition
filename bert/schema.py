# schema.py
from typing import (
    Callable,
    Dict,
    ItemsView,
    Iterable,
    Iterator,
    KeysView,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Union,
    ValuesView,
)

import torch

from enum import Enum

class FeatureType(Enum):
    """Type of Feature."""

    CATEGORICAL = "categorical"
    NUMERICAL = "numerical"

class FeatureHint(Enum):
    """Hint to algorithm about column."""

    ITEM_ID = "item_id"
    QUERY_ID = "query_id"
    RATING = "rating"
    TIMESTAMP = "timestamp"

class FeatureSource(Enum):
    """Name of DataFrame."""

    ITEM_FEATURES = "item_features"
    QUERY_FEATURES = "query_features"
    INTERACTIONS = "interactions"

class FeatureInfo:
    """
    Information about a feature.
    """

    def __init__(
        self,
        column: str,
        feature_type: FeatureType,
        feature_hint: Optional[FeatureHint] = None,
        feature_source: Optional[FeatureSource] = None,
        cardinality: Optional[int] = None,
    ) -> None:
        """
        :param column: name of feature.
        :param feature_type: type of feature.
        :param feature_hint: hint to models about feature
            (is timestamp, is rating, is query_id, is item_id),
            default: ``None``.
        :param feature_source: name of DataFrame feature came from,
            default: ``None``.
        :param cardinality: cardinality of categorical feature, required for ids columns,
            optional for others,
            default: ``None``.
        """
        self._column = column
        self._feature_type = feature_type
        self._feature_source = feature_source
        self._feature_hint = feature_hint

        if feature_type == FeatureType.NUMERICAL and cardinality:
            msg = "Cardinality is needed only with categorical feature_type."
            raise ValueError(msg)
        self._cardinality = cardinality

    @property
    def column(self) -> str:
        """
        :returns: the feature name.
        """
        return self._column

    @property
    def feature_type(self) -> FeatureType:
        """
        :returns: the type of feature.
        """
        return self._feature_type

    @property
    def feature_hint(self) -> Optional[FeatureHint]:
        """
        :returns: the feature hint.
        """
        return self._feature_hint

    @property
    def feature_source(self) -> Optional[FeatureSource]:
        """
        :returns: the name of source dataframe of feature.
        """
        return self._feature_source

    def _set_feature_source(self, source: FeatureSource) -> None:
        self._feature_source = source

    @property
    def cardinality(self) -> Optional[int]:
        """
        :returns: cardinality of the feature.
        """
        if self.feature_type != FeatureType.CATEGORICAL:
            msg = f"Can not get cardinality because feature_type of {self.column} column is not categorical."
            raise RuntimeError(msg)
        if hasattr(self, "_cardinality_callback") and self._cardinality is None:
            self._cardinality = self._cardinality_callback(self._column)
        return self._cardinality

    def _set_cardinality_callback(self, callback: Callable) -> None:
        self._cardinality_callback = callback

    def reset_cardinality(self) -> None:
        """
        Reset cardinality of the feature to None.
        """
        self._cardinality = None


class FeatureSchema(Mapping[str, FeatureInfo]):
    """
    Key-value like collection with information about all dataset features.
    """

    def __init__(self, features_list: Union[Sequence[FeatureInfo], FeatureInfo]) -> None:
        """
        :param features_list: list of feature infos.
        """
        features_list = [features_list] if not isinstance(features_list, Sequence) else features_list
        self._check_features_naming(features_list)
        self._features_schema = {feature.column: feature for feature in features_list}

    def copy(self) -> "FeatureSchema":
        """
        Creates a copy of all features.

        :returns: copy of the initial feature schema.
        """
        copy_features_list = list(self._features_schema.values())
        for feature in copy_features_list:
            feature.reset_cardinality()
        return FeatureSchema(copy_features_list)

    def subset(self, features_to_keep: Iterable[str]) -> "FeatureSchema":
        """
        Creates a subset of given features.

        :param features_to_keep: a sequence of feature columns
            in original schema to keep in subset.
        :returns: new feature schema of given features.
        """
        features: Set[FeatureInfo] = set()
        for feature_column in features_to_keep:
            if feature_column in self._features_schema:
                features.add(self._features_schema[feature_column])
        return FeatureSchema(list(features))

    def item(self) -> FeatureInfo:
        """
        :returns: extract a feature information from a schema.
        """
        if len(self._features_schema) > 1:
            msg = "Only one element feature schema can be converted to single feature"
            raise ValueError(msg)
        return next(iter(self._features_schema.values()))

    def items(self) -> ItemsView[str, FeatureInfo]:
        return self._features_schema.items()

    def keys(self) -> KeysView[str]:
        return self._features_schema.keys()

    def values(self) -> ValuesView[FeatureInfo]:
        return self._features_schema.values()

    def get(
        self,
        key: str,
        default: Optional[FeatureInfo] = None,
    ) -> Optional[FeatureInfo]:
        return self._features_schema.get(key, default)

    def __iter__(self) -> Iterator[str]:
        return iter(self._features_schema)

    def __contains__(self, feature_name: object) -> bool:
        return feature_name in self._features_schema

    def __len__(self) -> int:
        return len(self._features_schema)

    def __bool__(self) -> bool:
        return len(self._features_schema) > 0

    def __getitem__(self, feature_name: str) -> FeatureInfo:
        return self._features_schema[feature_name]

    def __eq__(self, other: object) -> bool:
        return self._features_schema == other

    def __ne__(self, other: object) -> bool:
        return self._features_schema != other

    def __add__(self, other: "FeatureSchema") -> "FeatureSchema":
        return FeatureSchema(list(self._features_schema.values()) + list(other._features_schema.values()))

    @property
    def all_features(self) -> Sequence[FeatureInfo]:
        """
        :returns: sequence of all features.
        """
        return list(self._features_schema.values())

    @property
    def categorical_features(self) -> "FeatureSchema":
        """
        :returns: sequence of categorical features in a schema.
        """
        return self.filter(feature_type=FeatureType.CATEGORICAL)

    @property
    def numerical_features(self) -> "FeatureSchema":
        """
        :returns: sequence of numerical features in a schema.
        """
        return self.filter(feature_type=FeatureType.NUMERICAL)

    @property
    def interaction_features(self) -> "FeatureSchema":
        """
        :returns: sequence of interaction features in a schema.
        """
        return (
            self.filter(feature_source=FeatureSource.INTERACTIONS)
            .drop(feature_hint=FeatureHint.ITEM_ID)
            .drop(feature_hint=FeatureHint.QUERY_ID)
        )

    @property
    def query_features(self) -> "FeatureSchema":
        """
        :returns: sequence of query features in a schema.
        """
        return self.filter(feature_source=FeatureSource.QUERY_FEATURES)

    @property
    def item_features(self) -> "FeatureSchema":
        """
        :returns: sequence of item features in a schema.
        """
        return self.filter(feature_source=FeatureSource.ITEM_FEATURES)

    @property
    def interactions_rating_features(self) -> "FeatureSchema":
        """
        :returns: sequence of interactions-rating features in a schema.
        """
        return self.filter(feature_source=FeatureSource.INTERACTIONS, feature_hint=FeatureHint.RATING)

    @property
    def interactions_timestamp_features(self) -> "FeatureSchema":
        """
        :returns: sequence of interactions-timestamp features in a schema.
        """
        return self.filter(feature_source=FeatureSource.INTERACTIONS, feature_hint=FeatureHint.TIMESTAMP)

    @property
    def columns(self) -> Sequence[str]:
        """
        :returns: list of all feature's column names.
        """
        return list(self._features_schema)

    @property
    def query_id_feature(self) -> FeatureInfo:
        """
        :returns: sequence of query id features in a schema.
        """
        return self.filter(feature_hint=FeatureHint.QUERY_ID).item()

    @property
    def item_id_feature(self) -> FeatureInfo:
        """
        :returns: sequence of item id features in a schema.
        """
        return self.filter(feature_hint=FeatureHint.ITEM_ID).item()

    @property
    def query_id_column(self) -> str:
        """
        :returns: query id column name.
        """
        return self.query_id_feature.column

    @property
    def item_id_column(self) -> str:
        """
        :returns: item id column name.
        """
        return self.item_id_feature.column

    @property
    def interactions_rating_column(self) -> Optional[str]:
        """
        :returns: interactions-rating column name.
        """
        interactions_rating_features = self.interactions_rating_features
        if not interactions_rating_features:
            return None
        return interactions_rating_features.item().column

    @property
    def interactions_timestamp_column(self) -> Optional[str]:
        """
        :returns: interactions-timestamp column name.
        """
        interactions_timestamp_features = self.interactions_timestamp_features
        if not interactions_timestamp_features:
            return None
        return interactions_timestamp_features.item().column

    def filter(
        self,
        column: Optional[str] = None,
        feature_hint: Optional[FeatureHint] = None,
        feature_source: Optional[FeatureSource] = None,
        feature_type: Optional[FeatureType] = None,
    ) -> "FeatureSchema":
        """Filter list by ``column``, ``feature_source``, ``feature_type`` and ``feature_hint``.

        :param column: Column name to filter by.
            default: ``None``.
        :param feature_hint: Feature hint to filter by.
            default: ``None``.
        :param feature_source: Feature source to filter by.
            default: ``None``.
        :param feature_type: Feature type to filter by.
            default: ``None``.

        :returns: new filtered feature schema.
        """
        filtered_features = self.all_features
        filter_functions = [self._name_filter, self._source_filter, self._type_filter, self._hint_filter]
        filter_parameters = [column, feature_source, feature_type, feature_hint]
        for filtration_func, filtration_param in zip(filter_functions, filter_parameters):
            filtered_features = list(
                filter(
                    lambda x: filtration_func(x, filtration_param),
                    filtered_features,
                )
            )

        return FeatureSchema(filtered_features)

    def drop(
        self,
        column: Optional[str] = None,
        feature_hint: Optional[FeatureHint] = None,
        feature_source: Optional[FeatureSource] = None,
        feature_type: Optional[FeatureType] = None,
    ) -> "FeatureSchema":
        """Drop features from list by ``column``, ``feature_source``, ``feature_type`` and ``feature_hint``.

        :param column: Column name to filter by.
            default: ``None``.
        :param feature_hint: Feature hint to filter by.
            default: ``None``.
        :param feature_source: Feature source to filter by.
            default: ``None``.
        :param feature_type: Feature type to filter by.
            default: ``None``.

        :returns: new filtered feature schema without selected features.
        """
        filtered_features = self.all_features
        filter_functions = [self._name_drop, self._source_drop, self._type_drop, self._hint_drop]
        filter_parameters = [column, feature_source, feature_type, feature_hint]
        for filtration_func, filtration_param in zip(filter_functions, filter_parameters):
            filtered_features = list(
                filter(
                    lambda x: filtration_func(x, filtration_param),
                    filtered_features,
                )
            )

        return FeatureSchema(filtered_features)

    @staticmethod
    def _name_filter(value: FeatureInfo, column: str) -> bool:
        return value.column == column if column else True

    @staticmethod
    def _source_filter(value: FeatureInfo, feature_source: FeatureSource) -> bool:
        return value.feature_source == feature_source if feature_source else True

    @staticmethod
    def _type_filter(value: FeatureInfo, feature_type: FeatureType) -> bool:
        return value.feature_type == feature_type if feature_type else True

    @staticmethod
    def _hint_filter(value: FeatureInfo, feature_hint: FeatureHint) -> bool:
        return value.feature_hint == feature_hint if feature_hint else True

    @staticmethod
    def _name_drop(value: FeatureInfo, column: str) -> bool:
        return value.column != column if column else True

    @staticmethod
    def _source_drop(value: FeatureInfo, feature_source: FeatureSource) -> bool:
        return value.feature_source != feature_source if feature_source else True

    @staticmethod
    def _type_drop(value: FeatureInfo, feature_type: FeatureType) -> bool:
        return value.feature_type != feature_type if feature_type else True

    @staticmethod
    def _hint_drop(value: FeatureInfo, feature_hint: FeatureHint) -> bool:
        return value.feature_hint != feature_hint if feature_hint else True

    def _check_features_naming(self, features_list: Sequence[FeatureInfo]) -> None:
        """
        Checks that all the columns have unique names except QUERY_ID and ITEM_ID columns.
        """
        unique_columns = set()
        duplicates = set()
        item_query_names: Dict[FeatureHint, List[str]] = {
            FeatureHint.ITEM_ID: [],
            FeatureHint.QUERY_ID: [],
        }
        for feature in features_list:
            if feature.feature_hint not in [FeatureHint.ITEM_ID, FeatureHint.QUERY_ID]:
                if feature.column in unique_columns:
                    duplicates.add(feature.column)
                else:
                    unique_columns.add(feature.column)
            else:
                item_query_names[feature.feature_hint] += [feature.column]

        if len(duplicates) > 0:
            msg = (
                "Features column names should be unique, exept ITEM_ID and QUERY_ID columns. "
                f"{duplicates} columns are not unique."
            )
            raise ValueError(msg)

        if len(item_query_names[FeatureHint.ITEM_ID]) > 1:
            msg = f"ITEM_ID must be present only once. Rename {item_query_names[FeatureHint.ITEM_ID]}"
            raise ValueError(msg)

        if len(item_query_names[FeatureHint.QUERY_ID]) > 1:
            msg = f"QUERY_ID must be present only once. Rename {item_query_names[FeatureHint.QUERY_ID]}"
            raise ValueError(msg)

# Alias
TensorMap = Mapping[str, torch.Tensor]
MutableTensorMap = Dict[str, torch.Tensor]


class TensorFeatureSource:
    """
    Describes source of a feature
    """

    def __init__(
        self,
        source: FeatureSource,
        column: str,
        index: Optional[int] = None,
    ) -> None:
        """
        :param source: feature source
        :param column: column name
        :param index: index of column in dataframe to get tensor
            directly, without mappings
        """
        self._column = column
        self._index = index
        self._source = source

    @property
    def source(self) -> FeatureSource:
        """
        :returns: feature source
        """
        return self._source

    @property
    def column(self) -> str:
        """
        :returns: column name
        """
        return self._column

    @property
    def index(self) -> Optional[int]:
        """
        :returns: provided index
        """
        return self._index


class TensorFeatureInfo:
    """
    Information about a tensor feature.
    """

    def __init__(
        self,
        name: str,
        feature_type: FeatureType,
        is_seq: bool = False,
        feature_hint: Optional[FeatureHint] = None,
        feature_sources: Optional[List[TensorFeatureSource]] = None,
        cardinality: Optional[int] = None,
        embedding_dim: Optional[int] = None,
        tensor_dim: Optional[int] = None,
    ) -> None:
        """
        :param name: name of feature.
        :param feature_type: type of feature.
        :param is_seq: flag that feature is sequential.
            default: ``False``.
        :param feature_hint: hint to models about feature
            (is timestamp, is rating, is query_id, is item_id),
            default: ``None``.
        :param feature_sources: columns names and DataFrames feature came from,
            default: ``None``.
        :param cardinality: cardinality of categorical feature, required for ids columns,
            optional for others,
            default: ``None``.
        :param embedding_dim: embedding dimensions of categorical feature,
            default: ``None``.
        :param tensor_dim: tensor dimensions of numerical feature,
            default: ``None``.
        """
        self._name = name
        self._feature_hint = feature_hint
        self._feature_sources = feature_sources
        self._is_seq = is_seq

        if not isinstance(feature_type, FeatureType):
            msg = "Unknown feature type"
            raise ValueError(msg)
        self._feature_type = feature_type

        if feature_type == FeatureType.NUMERICAL and (cardinality or embedding_dim):
            msg = "Cardinality and embedding dimensions are needed only with categorical feature type."
            raise ValueError(msg)
        self._cardinality = cardinality

        if feature_type == FeatureType.CATEGORICAL and tensor_dim:
            msg = "Tensor dimensions is needed only with numerical feature type."
            raise ValueError(msg)

        if feature_type == FeatureType.CATEGORICAL:
            default_embedding_dim = 64
            self._embedding_dim = embedding_dim or default_embedding_dim
        else:
            self._tensor_dim = tensor_dim

    @property
    def name(self) -> str:
        """
        :returns: The feature name.
        """
        return self._name

    @property
    def feature_type(self) -> FeatureType:
        """
        :returns: The type of feature.
        """
        return self._feature_type

    @property
    def feature_hint(self) -> Optional[FeatureHint]:
        """
        :returns: The feature hint.
        """
        return self._feature_hint

    def _set_feature_hint(self, hint: FeatureHint) -> None:
        self._feature_hint = hint

    @property
    def feature_sources(self) -> Optional[List[TensorFeatureSource]]:
        """
        :returns: List of sources feature came from.
        """
        return self._feature_sources

    def _set_feature_sources(self, sources: List[TensorFeatureSource]) -> None:
        self._feature_sources = sources

    @property
    def feature_source(self) -> Optional[TensorFeatureSource]:
        """
        :returns: Dataframe info of feature.
        """
        source = self.feature_sources
        if not source:
            return None

        if len(source) > 1:
            msg = "Only one element feature sources can be converted to single feature source."
            raise ValueError(msg)
        assert isinstance(self.feature_sources, list)
        return self.feature_sources[0]

    @property
    def is_seq(self) -> bool:
        """
        :returns: Flag that feature is sequential.
        """
        return self._is_seq

    @property
    def is_cat(self) -> bool:
        """
        :returns: Flag that feature is categorical.
        """
        return self.feature_type == FeatureType.CATEGORICAL

    @property
    def is_num(self) -> bool:
        """
        :returns: Flag that feature is numerical.
        """
        return self.feature_type == FeatureType.NUMERICAL

    @property
    def cardinality(self) -> Optional[int]:
        """
        :returns: Cardinality of the feature.
        """
        if self.feature_type != FeatureType.CATEGORICAL:
            msg = f"Can not get cardinality because feature type of {self.name} column is not categorical."
            raise RuntimeError(msg)
        return self._cardinality

    def _set_cardinality(self, cardinality: int) -> None:
        self._cardinality = cardinality

    @property
    def tensor_dim(self) -> Optional[int]:
        """
        :returns: Dimensions of the numerical feature.
        """
        if self.feature_type != FeatureType.NUMERICAL:
            msg = f"Can not get tensor dimensions because feature type of {self.name} feature is not numerical."
            raise RuntimeError(msg)
        return self._tensor_dim

    def _set_tensor_dim(self, tensor_dim: int) -> None:
        self._tensor_dim = tensor_dim

    @property
    def embedding_dim(self) -> Optional[int]:
        """
        :returns: Embedding dimensions of the feature.
        """
        if self.feature_type != FeatureType.CATEGORICAL:
            msg = f"Can not get embedding dimensions because feature type of {self.name} feature is not categorical."
            raise RuntimeError(msg)
        return self._embedding_dim

    def _set_embedding_dim(self, embedding_dim: int) -> None:
        self._embedding_dim = embedding_dim


class TensorSchema(Mapping[str, TensorFeatureInfo]):
    """
    Key-value like collection that stores tensor features
    """

    def __init__(self, features_list: Union[Sequence[TensorFeatureInfo], TensorFeatureInfo]) -> None:
        """
        :param features_list: list of tensor feature infos.
        """
        features_list = [features_list] if not isinstance(features_list, Sequence) else features_list
        self._tensor_schema = {feature.name: feature for feature in features_list}

    def subset(self, features_to_keep: Iterable[str]) -> "TensorSchema":
        """Creates a subset of given features.

        :param features_to_keep: A sequence of feature names
                in original schema to keep in subset.

        :returns: New tensor schema of given features.
        """
        features: Set[TensorFeatureInfo] = set()
        for feature_name in features_to_keep:
            features.add(self._tensor_schema[feature_name])
        return TensorSchema(list(features))

    def item(self) -> TensorFeatureInfo:
        """
        :returns: Extract single feature from a schema.
        """
        if len(self._tensor_schema) != 1:
            msg = "Only one element tensor schema can be converted to single feature"
            raise ValueError(msg)
        return next(iter(self._tensor_schema.values()))

    def items(self) -> ItemsView[str, TensorFeatureInfo]:
        return self._tensor_schema.items()

    def keys(self) -> KeysView[str]:
        return self._tensor_schema.keys()

    def values(self) -> ValuesView[TensorFeatureInfo]:
        return self._tensor_schema.values()

    def get(
        self,
        key: str,
        default: Optional[TensorFeatureInfo] = None,
    ) -> Optional[TensorFeatureInfo]:
        return self._tensor_schema.get(key, default)

    def __iter__(self) -> Iterator[str]:
        return iter(self._tensor_schema)

    def __contains__(self, feature_name: object) -> bool:
        return feature_name in self._tensor_schema

    def __len__(self) -> int:
        return len(self._tensor_schema)

    def __getitem__(self, feature_name: str) -> TensorFeatureInfo:
        return self._tensor_schema[feature_name]

    def __eq__(self, other: object) -> bool:
        return self._tensor_schema == other

    def __ne__(self, other: object) -> bool:
        return self._tensor_schema != other

    def __add__(self, other: "TensorSchema") -> "TensorSchema":
        return TensorSchema(list(self._tensor_schema.values()) + list(other._tensor_schema.values()))

    @property
    def all_features(self) -> Sequence[TensorFeatureInfo]:
        """
        :returns: Sequence of all features.
        """
        return list(self._tensor_schema.values())

    @property
    def categorical_features(self) -> "TensorSchema":
        """
        :returns: Sequence of categorical features in a schema.
        """
        return self.filter(feature_type=FeatureType.CATEGORICAL)

    @property
    def numerical_features(self) -> "TensorSchema":
        """
        :returns: Sequence of numerical features in a schema.
        """
        return self.filter(feature_type=FeatureType.NUMERICAL)

    @property
    def query_id_features(self) -> "TensorSchema":
        """
        :returns: Sequence of query id features in a schema.
        """
        return self.filter(feature_hint=FeatureHint.QUERY_ID)

    @property
    def item_id_features(self) -> "TensorSchema":
        """
        :returns: Sequence of item id features in a schema.
        """
        return self.filter(feature_hint=FeatureHint.ITEM_ID)

    @property
    def timestamp_features(self) -> "TensorSchema":
        """
        :returns: Sequence of timestamp features in a schema.
        """
        return self.filter(feature_hint=FeatureHint.TIMESTAMP)

    @property
    def rating_features(self) -> "TensorSchema":
        """
        :returns: Sequence of rating features in a schema.
        """
        return self.filter(feature_hint=FeatureHint.RATING)

    @property
    def sequential_features(self) -> "TensorSchema":
        """
        :returns: Sequence of sequential features in a schema.
        """
        return self.filter(is_seq=True)

    @property
    def names(self) -> Sequence[str]:
        """
        :returns: List of all feature's names.
        """
        return list(self._tensor_schema)

    @property
    def query_id_feature_name(self) -> Optional[str]:
        """
        :returns: Query id feature name.
        """
        query_id_features = self.query_id_features
        if not query_id_features:
            return None
        return query_id_features.item().name

    @property
    def item_id_feature_name(self) -> Optional[str]:
        """
        :returns: Item id feature name.
        """
        item_id_features = self.item_id_features
        if not item_id_features:
            return None
        return item_id_features.item().name

    @property
    def timestamp_feature_name(self) -> Optional[str]:
        """
        :returns: Timestamp feature name.
        """
        timestamp_features = self.timestamp_features
        if not timestamp_features:
            return None
        return timestamp_features.item().name

    @property
    def rating_feature_name(self) -> Optional[str]:
        """
        :returns: Rating feature name.
        """
        rating_features = self.rating_features
        if not rating_features:
            return None
        return rating_features.item().name

    def _get_object_args(self) -> Dict:
        """
        Returns list of features represented as dictionaries.
        """
        features = [
            {
                "name": feature.name,
                "feature_type": feature.feature_type.name,
                "is_seq": feature.is_seq,
                "feature_hint": feature.feature_hint.name if feature.feature_hint else None,
                "feature_sources": (
                    [{"source": x.source.name, "column": x.column, "index": x.index} for x in feature.feature_sources]
                    if feature.feature_sources
                    else None
                ),
                "cardinality": feature.cardinality if feature.feature_type == FeatureType.CATEGORICAL else None,
                "embedding_dim": feature.embedding_dim if feature.feature_type == FeatureType.CATEGORICAL else None,
                "tensor_dim": feature.tensor_dim if feature.feature_type == FeatureType.NUMERICAL else None,
            }
            for feature in self.all_features
        ]
        return features

    @classmethod
    def _create_object_by_args(cls, args: Dict) -> "TensorSchema":
        features_list = []
        for feature_data in args:
            feature_data["feature_sources"] = (
                [
                    TensorFeatureSource(source=FeatureSource[x["source"]], column=x["column"], index=x["index"])
                    for x in feature_data["feature_sources"]
                ]
                if feature_data["feature_sources"]
                else None
            )
            f_type = feature_data["feature_type"]
            f_hint = feature_data["feature_hint"]
            feature_data["feature_type"] = FeatureType[f_type] if f_type else None
            feature_data["feature_hint"] = FeatureHint[f_hint] if f_hint else None
            features_list.append(TensorFeatureInfo(**feature_data))
        return TensorSchema(features_list)

    def filter(
        self,
        name: Optional[str] = None,
        feature_hint: Optional[FeatureHint] = None,
        is_seq: Optional[bool] = None,
        feature_type: Optional[FeatureType] = None,
    ) -> "TensorSchema":
        """Filter list by ``name``, ``feature_type``, ``is_seq`` and ``feature_hint``.

        :param name: Feature name to filter by.
            default: ``None``.
        :param feature_hint: Feature hint to filter by.
            default: ``None``.
        :param feature_source: Feature source to filter by.
            default: ``None``.
        :param feature_type: Feature type to filter by.
            default: ``None``.

        :returns: New filtered feature schema.
        """
        filtered_features = self.all_features
        filter_functions = [self._name_filter, self._seq_filter, self._type_filter, self._hint_filter]
        filter_parameters = [name, is_seq, feature_type, feature_hint]
        for filtration_func, filtration_param in zip(filter_functions, filter_parameters):
            filtered_features = list(
                filter(
                    lambda x: filtration_func(x, filtration_param),
                    filtered_features,
                )
            )

        return TensorSchema(filtered_features)

    @staticmethod
    def _name_filter(value: TensorFeatureInfo, name: str) -> bool:
        return value.name == name if name else True

    @staticmethod
    def _seq_filter(value: TensorFeatureInfo, is_seq: bool) -> bool:
        return value.is_seq == is_seq if is_seq is not None else True

    @staticmethod
    def _type_filter(value: TensorFeatureInfo, feature_type: FeatureType) -> bool:
        return value.feature_type == feature_type if feature_type else True

    @staticmethod
    def _hint_filter(value: TensorFeatureInfo, feature_hint: FeatureHint) -> bool:
        return value.feature_hint == feature_hint if feature_hint else True