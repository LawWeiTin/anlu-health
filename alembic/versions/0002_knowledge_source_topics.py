"""Add explicit medical-topic labels to approved knowledge sources."""

import sqlalchemy as sa

from alembic import op

revision = "0002_knowledge_source_topics"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "knowledge_sources",
        sa.Column("topics", sa.JSON(), nullable=False, server_default="[]"),
    )


def downgrade() -> None:
    op.drop_column("knowledge_sources", "topics")
