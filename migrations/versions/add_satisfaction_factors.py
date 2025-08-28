"""add satisfaction factors

Revision ID: add_satisfaction_factors
Revises: ad0b59533d02
Create Date: 2025-01-15 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers
revision = 'add_satisfaction_factors'
down_revision = 'ad0b59533d02'
branch_labels = None
depends_on = None

def upgrade():
    # Add new satisfaction factor columns
    op.add_column('members', sa.Column('satis_wage', sa.Integer(), nullable=True, default=3))
    op.add_column('members', sa.Column('satis_stability', sa.Integer(), nullable=True, default=3))
    op.add_column('members', sa.Column('satis_growth', sa.Integer(), nullable=True, default=3))
    op.add_column('members', sa.Column('satis_task_content', sa.Integer(), nullable=True, default=3))
    op.add_column('members', sa.Column('satis_work_env', sa.Integer(), nullable=True, default=3))
    op.add_column('members', sa.Column('satis_work_time', sa.Integer(), nullable=True, default=3))
    op.add_column('members', sa.Column('satis_communication', sa.Integer(), nullable=True, default=3))
    op.add_column('members', sa.Column('satis_fair_eval', sa.Integer(), nullable=True, default=3))
    op.add_column('members', sa.Column('satis_welfare', sa.Integer(), nullable=True, default=3))
    
    # Add timestamp columns
    op.add_column('members', sa.Column('created_at', sa.DateTime(), nullable=True))
    op.add_column('members', sa.Column('updated_at', sa.DateTime(), nullable=True))
    
    # Drop old column (optional - uncomment if you want to remove it)
    # op.drop_column('members', 'satis_focus_key')

def downgrade():
    # Remove new columns
    op.drop_column('members', 'satis_wage')
    op.drop_column('members', 'satis_stability') 
    op.drop_column('members', 'satis_growth')
    op.drop_column('members', 'satis_task_content')
    op.drop_column('members', 'satis_work_env')
    op.drop_column('members', 'satis_work_time')
    op.drop_column('members', 'satis_communication')
    op.drop_column('members', 'satis_fair_eval')
    op.drop_column('members', 'satis_welfare')
    op.drop_column('members', 'created_at')
    op.drop_column('members', 'updated_at')
    
    # Restore old column (optional)
    # op.add_column('members', sa.Column('satis_focus_key', sa.String(50), nullable=True))