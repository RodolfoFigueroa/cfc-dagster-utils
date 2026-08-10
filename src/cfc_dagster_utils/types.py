from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Self, TypeVar, cast

if TYPE_CHECKING:
    import geopandas as gpd
    import pandas as pd

T = TypeVar("T")
DFType = TypeVar("DFType", "pd.DataFrame", "gpd.GeoDataFrame")

POSTGRES_METADATA_KEY = "postgres"


class PostgresWriteMode(StrEnum):
    """Supported publication behavior for PostgreSQL-backed assets."""

    CREATE = "create"
    APPEND = "append"
    REPLACE = "replace"
    TRUNCATE_INSERT = "truncate_insert"


class PostgresIndexMethod(StrEnum):
    """Index methods supported by :class:`PostgresTableSpec`."""

    BTREE = "btree"
    GIST = "gist"


def _tuple_of_strings(value: object, *, field_name: str) -> tuple[str, ...]:
    if (
        not isinstance(value, (list, tuple))
        or not value
        or not all(isinstance(item, str) and item for item in value)
    ):
        msg = f"{field_name} must be a non-empty sequence of non-empty strings"
        raise ValueError(msg)
    return tuple(item for item in value if isinstance(item, str))


@dataclass(frozen=True, slots=True)
class PostgresRelation:
    """A lightweight reference to a PostgreSQL relation."""

    name: str
    schema: str = "public"

    def __post_init__(self) -> None:
        if not self.name:
            msg = "PostgresRelation.name must not be empty"
            raise ValueError(msg)
        if not self.schema:
            msg = "PostgresRelation.schema must not be empty"
            raise ValueError(msg)

    @property
    def display_name(self) -> str:
        """Return a human-readable, non-SQL-qualified name."""
        return f"{self.schema}.{self.name}"

    def to_dict(self) -> dict[str, str]:
        """Serialize the relation to Dagster-compatible metadata."""
        return {"schema": self.schema, "name": self.name}

    @classmethod
    def from_dict(cls, value: Mapping[Any, object]) -> Self:
        """Deserialize a relation from metadata."""
        name = value.get("name")
        schema = value.get("schema", "public")
        if not isinstance(name, str) or not isinstance(schema, str):
            msg = "PostgresRelation metadata requires string 'name' and 'schema'"
            raise TypeError(msg)
        return cls(name=name, schema=schema)


@dataclass(frozen=True, slots=True)
class PostgresForeignKey:
    """A potentially composite PostgreSQL foreign-key declaration."""

    columns: tuple[str, ...]
    referenced_relation: PostgresRelation
    referenced_columns: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "columns",
            _tuple_of_strings(self.columns, field_name="columns"),
        )
        object.__setattr__(
            self,
            "referenced_columns",
            _tuple_of_strings(
                self.referenced_columns,
                field_name="referenced_columns",
            ),
        )
        if len(self.columns) != len(self.referenced_columns):
            msg = "Foreign-key columns and referenced_columns must have equal length"
            raise ValueError(msg)

    def to_dict(self) -> dict[str, Any]:
        """Serialize this declaration to metadata."""
        return {
            "columns": list(self.columns),
            "referenced_relation": self.referenced_relation.to_dict(),
            "referenced_columns": list(self.referenced_columns),
        }

    @classmethod
    def from_dict(cls, value: Mapping[Any, object]) -> Self:
        """Deserialize this declaration from metadata."""
        relation = value.get("referenced_relation")
        if not isinstance(relation, Mapping):
            msg = "Foreign-key metadata requires 'referenced_relation'"
            raise TypeError(msg)
        return cls(
            columns=_tuple_of_strings(value.get("columns"), field_name="columns"),
            referenced_relation=PostgresRelation.from_dict(relation),
            referenced_columns=_tuple_of_strings(
                value.get("referenced_columns"),
                field_name="referenced_columns",
            ),
        )


@dataclass(frozen=True, slots=True)
class PostgresIndex:
    """A PostgreSQL index declaration."""

    columns: tuple[str, ...]
    method: PostgresIndexMethod = PostgresIndexMethod.BTREE
    unique: bool = False
    name: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "columns",
            _tuple_of_strings(self.columns, field_name="columns"),
        )
        if isinstance(self.method, str):
            object.__setattr__(self, "method", PostgresIndexMethod(self.method))
        if self.name == "":
            msg = "PostgresIndex.name must be non-empty when provided"
            raise TypeError(msg)

    def to_dict(self) -> dict[str, Any]:
        """Serialize this declaration to metadata."""
        return {
            "columns": list(self.columns),
            "method": self.method.value,
            "unique": self.unique,
            "name": self.name,
        }

    @classmethod
    def from_dict(cls, value: Mapping[Any, object]) -> Self:
        """Deserialize this declaration from metadata."""
        method = value.get("method", PostgresIndexMethod.BTREE.value)
        unique = value.get("unique", False)
        name = value.get("name")
        if not isinstance(method, str) or not isinstance(unique, bool):
            msg = "Invalid index method or unique flag"
            raise TypeError(msg)
        if name is not None and not isinstance(name, str):
            msg = "Index name must be a string or null"
            raise TypeError(msg)
        return cls(
            columns=_tuple_of_strings(value.get("columns"), field_name="columns"),
            method=PostgresIndexMethod(method),
            unique=unique,
            name=name,
        )


@dataclass(frozen=True, slots=True)
class PostgresTableSpec:
    """Declarative storage contract for a PostgreSQL-backed Dagster asset."""

    relation: PostgresRelation
    write_mode: PostgresWriteMode = PostgresWriteMode.CREATE
    primary_key: tuple[str, ...] = ()
    foreign_keys: tuple[PostgresForeignKey, ...] = field(default_factory=tuple)
    indexes: tuple[PostgresIndex, ...] = field(default_factory=tuple)
    geometry_column: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.relation, PostgresRelation):
            msg = "relation must be a PostgresRelation"
            raise TypeError(msg)
        if isinstance(self.write_mode, str):
            object.__setattr__(self, "write_mode", PostgresWriteMode(self.write_mode))
        if self.primary_key:
            object.__setattr__(
                self,
                "primary_key",
                _tuple_of_strings(self.primary_key, field_name="primary_key"),
            )
        object.__setattr__(self, "foreign_keys", tuple(self.foreign_keys))
        object.__setattr__(self, "indexes", tuple(self.indexes))
        if not all(
            isinstance(foreign_key, PostgresForeignKey)
            for foreign_key in self.foreign_keys
        ):
            msg = "foreign_keys must contain PostgresForeignKey values"
            raise TypeError(msg)
        if not all(isinstance(index, PostgresIndex) for index in self.indexes):
            msg = "indexes must contain PostgresIndex values"
            raise TypeError(msg)
        if self.geometry_column == "":
            msg = "geometry_column must be non-empty when provided"
            raise ValueError(msg)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the table contract to plain metadata values."""
        return {
            "relation": self.relation.to_dict(),
            "write_mode": self.write_mode.value,
            "primary_key": list(self.primary_key),
            "foreign_keys": [key.to_dict() for key in self.foreign_keys],
            "indexes": [index.to_dict() for index in self.indexes],
            "geometry_column": self.geometry_column,
        }

    def to_dagster_metadata(self) -> dict[str, dict[str, Any]]:
        """Return metadata ready for a Dagster asset or output definition."""
        return {POSTGRES_METADATA_KEY: self.to_dict()}

    @classmethod
    def from_dict(cls, value: Mapping[Any, object]) -> Self:
        """Deserialize a table contract from plain metadata values."""
        relation = value.get("relation")
        if not isinstance(relation, Mapping):
            msg = "PostgresTableSpec metadata requires a 'relation' mapping"
            raise TypeError(msg)

        write_mode = value.get("write_mode", PostgresWriteMode.CREATE.value)
        primary_key = value.get("primary_key", [])
        foreign_keys = value.get("foreign_keys", [])
        indexes = value.get("indexes", [])
        geometry_column = value.get("geometry_column")
        if not isinstance(write_mode, str):
            msg = "write_mode must be a string"
            raise TypeError(msg)
        if not isinstance(primary_key, (list, tuple)):
            msg = "primary_key must be a sequence"
            raise TypeError(msg)
        if not isinstance(foreign_keys, (list, tuple)) or not all(
            isinstance(item, Mapping) for item in foreign_keys
        ):
            msg = "foreign_keys must be a sequence of mappings"
            raise ValueError(msg)
        if not isinstance(indexes, (list, tuple)) or not all(
            isinstance(item, Mapping) for item in indexes
        ):
            msg = "indexes must be a sequence of mappings"
            raise ValueError(msg)
        if geometry_column is not None and not isinstance(geometry_column, str):
            msg = "geometry_column must be a string or null"
            raise ValueError(msg)

        return cls(
            relation=PostgresRelation.from_dict(relation),
            write_mode=PostgresWriteMode(write_mode),
            primary_key=(
                _tuple_of_strings(primary_key, field_name="primary_key")
                if primary_key
                else ()
            ),
            foreign_keys=tuple(
                PostgresForeignKey.from_dict(cast("Mapping[Any, object]", item))
                for item in foreign_keys
            ),
            indexes=tuple(
                PostgresIndex.from_dict(cast("Mapping[Any, object]", item))
                for item in indexes
            ),
            geometry_column=geometry_column,
        )

    @classmethod
    def from_dagster_metadata(cls, metadata: Mapping[str, Any]) -> Self:
        """Read a table contract from Dagster definition metadata."""
        value = metadata.get(POSTGRES_METADATA_KEY)
        if not isinstance(value, Mapping):
            msg = (
                "PostgreSQL-backed assets require metadata generated by "
                "PostgresTableSpec.to_dagster_metadata()"
            )
            raise TypeError(msg)
        return cls.from_dict(value)
