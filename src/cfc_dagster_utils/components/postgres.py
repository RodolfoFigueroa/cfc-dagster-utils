from cfc_dagster_utils._optional import raise_optional_dependency_error

try:
    import dagster as dg
    from pydantic import field_validator
except ModuleNotFoundError as error:
    raise_optional_dependency_error(
        error,
        import_name="dagster",
        dependency_name="dagster",
        extra="postgres",
    )

from cfc_dagster_utils.managers.postgres import PostgresIOManager
from cfc_dagster_utils.resources import PostgresResource


class PostgresConnectionComponent(dg.Component, dg.Resolvable, dg.Model):
    """Configure a PostgreSQL resource and output-only IO manager."""

    host: str
    port: int
    user: str
    password: str
    database: str
    resource_key: str = "postgres_resource"
    io_manager_key: str = "postgres_manager"
    batch_size: int = 1000

    @field_validator("batch_size", mode="before")
    @classmethod
    def validate_batch_size(cls, value: object) -> int:
        return PostgresIOManager.validate_batch_size(value)

    def model_post_init(self, context: object, /) -> None:  # noqa: ARG002
        if self.resource_key == self.io_manager_key:
            msg = "resource_key and io_manager_key must be different"
            raise ValueError(msg)

    def create_resource(self) -> PostgresResource:
        """Create the resource represented by this connection configuration."""
        return PostgresResource(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            database=self.database,
        )

    def build_defs(self, context: dg.ComponentLoadContext) -> dg.Definitions:  # noqa: ARG002
        resource = self.create_resource()
        return dg.Definitions(
            resources={
                self.resource_key: resource,
                self.io_manager_key: PostgresIOManager(
                    postgres_resource=resource, batch_size=self.batch_size
                ),
            },
        )
