"""Add curated index terms, a stored lexical document, and PostgreSQL full-text search."""

import sqlalchemy as sa

from alembic import op

revision = "0003_hybrid_search_text"
down_revision = "0002_knowledge_source_topics"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "knowledge_sources",
        sa.Column("keywords", sa.JSON(), nullable=False, server_default="[]"),
    )
    op.add_column(
        "knowledge_chunks",
        sa.Column("search_text", sa.Text(), nullable=False, server_default=""),
    )
    op.execute("UPDATE knowledge_chunks SET search_text = content WHERE search_text = ''")
    if op.get_bind().dialect.name == "postgresql":
        op.execute(
            "CREATE INDEX ix_knowledge_chunks_search_text_fts "
            "ON knowledge_chunks USING gin (to_tsvector('simple', search_text))"
        )


def downgrade() -> None:
    if op.get_bind().dialect.name == "postgresql":
        op.execute("DROP INDEX IF EXISTS ix_knowledge_chunks_search_text_fts")
    op.drop_column("knowledge_chunks", "search_text")
    op.drop_column("knowledge_sources", "keywords")
