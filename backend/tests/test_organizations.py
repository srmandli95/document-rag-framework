from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.auth.dependencies import get_data_scope_id, require_data_scope_editor
from app.db.database import Base
from app.models.organization import Organization, OrganizationMembership
from app.models.user import User
from app.services.organization_service import (
    add_organization_member,
    create_organization,
    get_membership,
    list_user_organizations,
)


def test_organization_membership_and_active_scope():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(
        bind=engine,
        tables=[User.__table__, Organization.__table__, OrganizationMembership.__table__],
    )
    db = sessionmaker(bind=engine)()
    owner = User(email="owner@example.com", auth_provider="local", is_active=True)
    member = User(email="member@example.com", auth_provider="local", is_active=True)
    db.add_all([owner, member])
    db.commit()

    organization = create_organization(db, name="Example Org", owner_user_id=owner.id)
    add_organization_member(
        db,
        organization_id=organization.id,
        user_id=member.id,
        role="member",
    )

    assert get_membership(db, user_id=owner.id, organization_id=organization.id).role == "admin"
    assert list_user_organizations(db, user_id=member.id)[0][0].id == organization.id

    owner.active_organization_id = organization.id
    owner.organization_role = "admin"
    assert get_data_scope_id(owner) == organization.id
    require_data_scope_editor(owner)

    member.active_organization_id = organization.id
    member.organization_role = "member"
    with pytest.raises(HTTPException) as exc_info:
        require_data_scope_editor(member)
    assert exc_info.value.status_code == 403

    db.close()
import pytest
from fastapi import HTTPException
