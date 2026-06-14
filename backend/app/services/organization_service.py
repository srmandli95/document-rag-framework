from sqlalchemy.orm import Session

from app.models.organization import Organization, OrganizationMembership


VALID_ORGANIZATION_ROLES = {"admin", "editor", "member"}


def create_organization(
    db: Session,
    *,
    name: str,
    owner_user_id: str,
    external_id: str | None = None,
) -> Organization:
    clean_name = name.strip()
    if not clean_name:
        raise ValueError("Organization name is required")

    organization = Organization(name=clean_name, external_id=external_id)
    db.add(organization)
    db.flush()
    db.add(
        OrganizationMembership(
            organization_id=organization.id,
            user_id=owner_user_id,
            role="admin",
        )
    )
    db.commit()
    db.refresh(organization)
    return organization


def get_membership(
    db: Session,
    *,
    user_id: str,
    organization_id: str,
) -> OrganizationMembership | None:
    return (
        db.query(OrganizationMembership)
        .filter(
            OrganizationMembership.user_id == user_id,
            OrganizationMembership.organization_id == organization_id,
        )
        .first()
    )


def list_user_organizations(
    db: Session,
    *,
    user_id: str,
) -> list[tuple[Organization, OrganizationMembership]]:
    return (
        db.query(Organization, OrganizationMembership)
        .join(
            OrganizationMembership,
            OrganizationMembership.organization_id == Organization.id,
        )
        .filter(OrganizationMembership.user_id == user_id)
        .order_by(Organization.name.asc())
        .all()
    )


def add_organization_member(
    db: Session,
    *,
    organization_id: str,
    user_id: str,
    role: str,
) -> OrganizationMembership:
    clean_role = role.strip().lower()
    if clean_role not in VALID_ORGANIZATION_ROLES:
        raise ValueError("Role must be admin, editor, or member")
    existing = get_membership(
        db,
        user_id=user_id,
        organization_id=organization_id,
    )
    if existing:
        existing.role = clean_role
        membership = existing
    else:
        membership = OrganizationMembership(
            organization_id=organization_id,
            user_id=user_id,
            role=clean_role,
        )
        db.add(membership)
    db.commit()
    db.refresh(membership)
    return membership
